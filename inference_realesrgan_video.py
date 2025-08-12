import argparse
import cv2
import glob
import mimetypes
import numpy as np
import os
import shutil
import subprocess
import torch
import re
import threading
import queue
from os import path as osp
from tqdm import tqdm

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# ffmpeg-python (must be installed in your venv)
try:
    import ffmpeg
except ImportError as e:
    raise ImportError("Missing 'ffmpeg-python'. Install it in your venv: pip install ffmpeg-python") from e


# ----------------------------- media probing ----------------------------- #
def get_video_meta_info(path):
    """Return dict with width/height/fps/nb_frames and audio meta (or None)."""
    import json
    from fractions import Fraction

    probe = ffmpeg.probe(path)
    vstreams = [s for s in probe['streams'] if s.get('codec_type') == 'video']
    if not vstreams:
        raise RuntimeError('No video stream found')
    vs = vstreams[0]

    width = int(vs['width'])
    height = int(vs['height'])

    # fps: prefer avg_frame_rate, else r_frame_rate
    fps_str = (vs.get('avg_frame_rate') or vs.get('r_frame_rate') or '0/1')
    try:
        fps = float(Fraction(fps_str))
    except Exception:
        fps = 0.0

    # frames (nb_frames often missing on mkv)
    nb = vs.get('nb_frames')
    if not nb or nb == '0':
        p = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames', '1',
             '-show_entries', 'stream=nb_read_frames,nb_frames,avg_frame_rate,duration',
             '-of', 'json', path],
            capture_output=True, text=True
        )
        if p.returncode == 0 and p.stdout.strip():
            try:
                data = json.loads(p.stdout)
                s = data['streams'][0]
                nb = s.get('nb_read_frames') or s.get('nb_frames') or nb
                fps2 = s.get('avg_frame_rate')
                if fps2:
                    try:
                        from fractions import Fraction as F
                        fps = float(F(fps2))
                    except Exception:
                        pass
                if (not nb or nb == '0') and fps > 0:
                    dur = s.get('duration') or vs.get('duration') or probe.get('format', {}).get('duration')
                    try:
                        dur = float(dur)
                        nb = int(dur * fps + 0.5)
                    except Exception:
                        pass
            except Exception:
                pass
    nb_frames = int(nb) if nb and nb != 'N/A' else 0

    # audio (always include 'audio' key)
    audio_info = None
    astreams = [s for s in probe['streams'] if s.get('codec_type') == 'audio']
    if astreams:
        a = astreams[0]
        try:
            audio_info = {
                'index': a.get('index', 1),
                'codec_name': a.get('codec_name'),
                'sample_rate': int(a.get('sample_rate') or 0),
                'channels': int(a.get('channels') or 0),
            }
        except Exception:
            audio_info = None

    return {
        'width': width, 'height': height,
        'fps': fps, 'nb_frames': nb_frames,
        'audio': audio_info,
    }


def get_sub_video(args, num_process, process_idx):
    if num_process == 1:
        return args.input
    meta = get_video_meta_info(args.input)
    fps = meta['fps'] if meta['fps'] and meta['fps'] > 0 else (args.fps or 24)
    duration = int((meta['nb_frames'] / fps) if fps else 0)
    part_time = max(1, duration // num_process) if duration else 0
    print(f'duration: {duration}, part_time: {part_time}')
    os.makedirs(osp.join(args.output, f'{args.video_name}_inp_tmp_videos'), exist_ok=True)
    out_path = osp.join(args.output, f'{args.video_name}_inp_tmp_videos', f'{process_idx:03d}.mp4')
    # slice by time; last chunk open-ended
    cmd = [
        args.ffmpeg_bin, f'-i "{args.input}"',
        '-ss', f'{part_time * process_idx}',
        f'-to {part_time * (process_idx + 1)}' if (part_time and process_idx != num_process - 1) else '',
        '-async', '1', f'"{out_path}"', '-y'
    ]
    print(' '.join(cmd))
    subprocess.call(' '.join(cmd), shell=True)
    return out_path


# ----------------------------- IO helpers ----------------------------- #
class DecodeThread(threading.Thread):
    """Background decoder reading rawvideo to numpy frames and pushing to a queue."""
    def __init__(self, video_path, width, height, ffmpeg_bin, max_queue=8):
        super().__init__(daemon=True)
        self.width, self.height = width, height
        self.frame_bytes = width * height * 3
        self.q = queue.Queue(max_queue)
        self.proc = (
            ffmpeg
            .input(video_path)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', loglevel='error')
            .run_async(pipe_stdin=True, pipe_stdout=True, cmd=ffmpeg_bin)
        )

    def run(self):
        try:
            while True:
                buf = self.proc.stdout.read(self.frame_bytes)
                if not buf:
                    break
                # reuse-able pinned-ish buffer path: use numpy + later .pin_memory() on torch
                frame = np.frombuffer(buf, np.uint8).reshape([self.height, self.width, 3])
                self.q.put(frame.copy())  # copy so next read doesn't overwrite
        finally:
            try:
                self.proc.stdin.close()
                self.proc.wait()
            except Exception:
                pass
            self.q.put(None)  # sentinel

    def get(self):
        return self.q.get()


class Reader:
    def __init__(self, args, total_workers=1, worker_idx=0, queue_size=8):
        self.args = args
        input_type = mimetypes.guess_type(args.input)[0]
        self.input_type = 'folder' if input_type is None else input_type
        self.paths = []  # for image & folder types
        self.audio = None
        self.input_fps = None
        self.src_for_audio = None
        self.idx = 0
        self.decode_thread = None

        if self.input_type.startswith('video'):
            video_path = get_sub_video(args, total_workers, worker_idx)
            self.src_for_audio = video_path
            meta = get_video_meta_info(video_path)
            self.width = meta['width']
            self.height = meta['height']
            self.input_fps = meta['fps']
            self.audio = meta['audio']
            self.nb_frames = meta['nb_frames']
            # background decode
            self.decode_thread = DecodeThread(video_path, self.width, self.height, args.ffmpeg_bin, max_queue=queue_size)
            self.decode_thread.start()
        else:
            if self.input_type.startswith('image'):
                self.paths = [args.input]
            else:
                paths = sorted(glob.glob(os.path.join(args.input, '*')))
                tot_frames = len(paths)
                num_frame_per_worker = tot_frames // total_workers + (1 if tot_frames % total_workers else 0)
                self.paths = paths[num_frame_per_worker * worker_idx:num_frame_per_worker * (worker_idx + 1)]
            self.nb_frames = len(self.paths)
            assert self.nb_frames > 0, 'empty folder'
            from PIL import Image
            tmp_img = Image.open(self.paths[0])
            self.width, self.height = tmp_img.size

    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        if self.args.fps is not None:
            return self.args.fps
        elif self.input_fps is not None and self.input_fps > 0:
            return self.input_fps
        return 24

    def get_audio(self):
        return self.audio

    def get_audio_source(self):
        return self.src_for_audio if self.audio else None

    def __len__(self):
        return self.nb_frames

    def get_frame(self):
        if self.input_type.startswith('video'):
            return self.decode_thread.get()
        else:
            if self.idx >= self.nb_frames:
                return None
            img = cv2.imread(self.paths[self.idx])
            self.idx += 1
            return img

    def close(self):
        # decode thread auto-joins when sentinel consumed; nothing to do
        pass


class Writer:
    def __init__(self, args, audio_src, height, width, video_save_path, fps):
        out_width, out_height = int(width * args.outscale), int(height * args.outscale)
        if out_height > 2160:
            print('You are generating >4K; consider reducing -s for speed.')

        v_in = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}', framerate=fps)
        a_in = ffmpeg.input(audio_src).audio if audio_src else None

        if args.encoder in ('h264_vaapi', 'hevc_vaapi'):
            # Upload raw frames to VAAPI and encode on GPU
            v_hw = v_in.filter('format', 'nv12').filter('hwupload', derive_device='vaapi')  # <— let ffmpeg create a VAAPI device
            if a_in:
                out = ffmpeg.output(
                    v_hw, a_in, video_save_path,
                    vcodec=args.encoder, qp=args.qp,
                    bf=0,                 # disable B-frames → less buffering, faster
                    g=120,                # GOP length (tweak 60–240); smaller can reduce latency
                    acodec='copy', loglevel='error'
                )
            else:
                out = ffmpeg.output(
                    v_hw, video_save_path,
                    vcodec=args.encoder, qp=args.qp,
                    bf=0,                 # disable B-frames → less buffering, faster
                    g=120,                # GOP length (tweak 60–240); smaller can reduce latency
                    loglevel='error'
                )
            out = out.global_args('-vaapi_device', getattr(args, 'vaapi_device', '/dev/dri/renderD128'))
        else:
            # CPU x264, allow preset/CRF
            if a_in:
                out = ffmpeg.output(
                    v_in, a_in, video_save_path,
                    vcodec='libx264', pix_fmt='yuv420p',
                    preset=args.preset, crf=args.crf,
                    bf=0,                 # disable B-frames → less buffering, faster
                    g=120,                # GOP length (tweak 60–240); smaller can reduce latency
                    acodec='copy', loglevel='error'
                )
            else:
                out = ffmpeg.output(
                    v_in, video_save_path,
                    vcodec='libx264', pix_fmt='yuv420p',
                    preset=args.preset, crf=args.crf,
                    bf=0,                 # disable B-frames → less buffering, faster
                    g=120,                # GOP length (tweak 60–240); smaller can reduce latency
                    loglevel='error'
                )

        self.stream_writer = out.overwrite_output().run_async(
            pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin
        )

    def write_frame(self, frame_np: np.ndarray):
        self.stream_writer.stdin.write(frame_np.astype(np.uint8).tobytes())

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()


# ----------------------------- model inference helpers ----------------------------- #
def _load_state_dict_any(path):
    # Try safetensors first (if installed), else torch.load
    try:
        import safetensors.torch as st
        return st.load_file(path)
    except Exception:
        pass
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict):
        for key in ('params_ema', 'params', 'state_dict'):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj  # some checkpoints are just a flat state_dict


def infer_srvgg_from_ckpt(ckpt_path, out_nc=3):
    """Infer SRVGGNetCompact(num_feat, num_conv, upscale) from checkpoint tensors."""
    sd = _load_state_dict_any(ckpt_path)
    if not isinstance(sd, dict):
        raise RuntimeError("Checkpoint did not contain a state_dict-like mapping")

    # collect only 4D conv weights that look like body.N.weight (or trunk.N.weight)
    body = []
    pat = re.compile(r'(?:^|\.)(?:body|trunk)\.(\d+)\.weight$')
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor) or v.ndim != 4:
            continue
        m = pat.search(k)
        if m:
            body.append((int(m.group(1)), v.shape, k))

    if not body:
        any4d = [(k, v.shape) for k, v in sd.items() if isinstance(v, torch.Tensor) and v.ndim == 4]
        if not any4d:
            raise RuntimeError("No 4D convolution weights found in checkpoint")
        candidates = [(k, s) for k, s in any4d if s[0] in (out_nc * 4, out_nc * 9, out_nc * 16)]
        _, last_shape = (candidates[-1] if candidates else any4d[-1])
        out_ch, in_ch, _, _ = last_shape
        upscale = int(round(((out_ch // out_nc) ** 0.5)))
        upscale = upscale if upscale in (2, 3, 4) else 2
        # num_feat from conv_first if present, else from any square conv
        num_feat = None
        w = sd.get('conv_first.weight')
        if isinstance(w, torch.Tensor) and w.ndim == 4:
            num_feat = int(w.shape[0])
        if num_feat is None:
            for _, s in any4d:
                if s[0] == s[1]:
                    num_feat = int(s[0]); break
        num_feat = num_feat or 64
        num_conv = 32
        return int(num_feat), int(num_conv), int(upscale)

    # normal path using body.N.weight
    last_idx, last_shape, _ = max(body, key=lambda x: x[0])
    out_ch, in_ch, _, _ = last_shape
    upscale = int(round(((out_ch // out_nc) ** 0.5)))
    if upscale not in (2, 3, 4):
        upscale = 2

    # prefer conv_first for num_feat; else first square conv in body
    num_feat = None
    w = sd.get('conv_first.weight')
    if isinstance(w, torch.Tensor) and w.ndim == 4:
        num_feat = int(w.shape[0])
    if num_feat is None:
        for _, s, _ in sorted(body, key=lambda x: x[0]):
            if s[0] == s[1]:
                num_feat = int(s[0]); break
    num_feat = num_feat or 64

    # count body convs with square num_feat before the final (to 12/27/48)
    num_conv = sum(1 for idx, s, _ in body if idx < last_idx and s[0] == s[1] == num_feat)
    if num_conv == 0:
        num_conv = min(64, max(8, last_idx))

    return int(num_feat), int(num_conv), int(upscale)


# ----------------------------- batch helpers ----------------------------- #
def preprocess_rgb_batch(frames_np: list):
    """RGB uint8 list -> torch half RGB normalized NHWC -> NCHW channels_last."""
    arr = np.stack(frames_np, axis=0).astype(np.float32) / 255.0   # already RGB from pipe
    t = torch.from_numpy(arr).to(memory_format=torch.channels_last)  # NHWC
    t = t.permute(0, 3, 1, 2)  # NCHW
    return t


def postprocess_to_bgr_batch(out_t: torch.Tensor):
    """torch NCHW [0,1] RGB -> list of BGR uint8 frames."""
    out = out_t.permute(0, 2, 3, 1).clamp_(0, 1).mul_(255.0).round_().byte().cpu().numpy()  # NHWC RGB
    return [cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in out]


# ----------------------------- main inference ----------------------------- #
def build_model_and_path(args):
    """Resolve model_path and create model (or leave None to be inferred). Returns (model, netscale, model_path)."""
    args.model_name = args.model_name.split('.pth')[0]
    model = None
    netscale = 2
    file_url = []

    # Official presets (keep for convenience)
    if args.model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4); netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif args.model_name == 'RealESRNet_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4); netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4); netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif args.model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2); netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif args.model_name == 'realesr-animevideov3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'); netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif args.model_name == 'realesr-general-x4v3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'); netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]

    # determine model path
    if args.model_path:
        model_path = args.model_path
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
    else:
        model_path = os.path.join('weights', args.model_name + '.pth')
        if not os.path.isfile(model_path):
            if file_url:
                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                for url in file_url:
                    model_path = load_file_from_url(url=url, model_dir=os.path.join(ROOT_DIR, 'weights'),
                                                    progress=True, file_name=None)
            else:
                raise FileNotFoundError(
                    f"Model weight not found at {model_path}. Provide --model_path or put the file in ./weights/"
                )

    # build model if custom
    if model is None:
        num_feat, num_conv, inferred_scale = infer_srvgg_from_ckpt(model_path, out_nc=3)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=num_feat, num_conv=num_conv,
                                upscale=inferred_scale, act_type='prelu')
        netscale = inferred_scale

    return model, netscale, model_path


def optimize_model_for_rocm(model, compile_ok=True):
    """channels_last + optional torch.compile with autotune."""
    model = model.to(memory_format=torch.channels_last)
    if compile_ok:
        try:
            import torch._inductor.config as ic
            ic.max_autotune = True
            ic.coordinate_descent_tuner = True
        except Exception:
            pass
        try:
            model = torch.compile(model, dynamic=True)
        except Exception:
            pass
    return model


def run_batch_direct(model, frames, device):
    """Run true micro-batch when tiling is disabled. Returns list of BGR frames."""
    # preprocess (CPU), then pin and H2D non_blocking
    t = preprocess_rgb_batch(frames)  # NCHW
    t = t.half() if torch.cuda.is_available() else t.float()
    if t.device.type == 'cpu':
        try:
            t = t.pin_memory()
        except Exception:
            pass
    t = t.to(device, non_blocking=True)
    with torch.no_grad():
        out = model(t)
    return postprocess_to_bgr_batch(out)


def inference_video(args, video_save_path, device=None, total_workers=1, worker_idx=0):
    # build model + path
    model, netscale, model_path = build_model_and_path(args)

    # upsampler (for tiling path or generic fallback)
    upsampler = RealESRGANer(
        scale=netscale, model_path=model_path, dni_weight=None, model=model,
        tile=args.tile, tile_pad=args.tile_pad, pre_pad=args.pre_pad,
        half=not args.fp32, device=device,
    )

    # Optimize model
    upsampler.model = optimize_model_for_rocm(upsampler.model, compile_ok=(not args.no_compile))

    # GFPGAN optional
    face_enhancer = None
    if 'anime' in args.model_name and args.face_enhance:
        print('face_enhance is not supported in anime models; disabling it.')
        args.face_enhance = False
    if args.face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale, arch='clean', channel_multiplier=2, bg_upsampler=upsampler
        )

    # Prepare IO
    reader = Reader(args, total_workers, worker_idx, queue_size=args.queue_size)
    audio_src = reader.get_audio_source()
    height, width = reader.get_resolution()
    fps = reader.get_fps()
    writer = Writer(args, audio_src, height, width, video_save_path, fps)

    # Main loop (decode/compute overlapped via background thread)
    pbar = tqdm(total=len(reader), unit='frame', desc='inference')
    batch_buf = []
    while True:
        frame = reader.get_frame()
        if frame is None:
            # flush remaining
            if batch_buf:
                if args.tile == 0 and args.batch > 1 and args.outscale == netscale and face_enhancer is None:
                    outs = run_batch_direct(upsampler.model, batch_buf, device)
                    for o in outs:
                        writer.write_frame(o)
                else:
                    for f in batch_buf:
                        try:
                            if face_enhancer is not None:
                                _, _, out = face_enhancer.enhance(f, has_aligned=False, only_center_face=False, paste_back=True)
                            else:
                                out, _ = upsampler.enhance(f, outscale=args.outscale)
                        except RuntimeError as e:
                            print('Error', e)
                        else:
                            writer.write_frame(out)
                pbar.update(len(batch_buf))
                batch_buf.clear()
            break

        batch_buf.append(frame)

        # If batching is enabled and tiling is OFF and no face enhance, do real micro-batch
        if args.tile == 0 and args.batch > 1 and len(batch_buf) >= args.batch and args.outscale == netscale and face_enhancer is None:
            outs = run_batch_direct(upsampler.model, batch_buf, device)
            for o in outs:
                writer.write_frame(o)
            pbar.update(len(batch_buf))
            batch_buf.clear()

        # Fallback per-frame path (tiling, or faces, or outscale != netscale)
        elif args.tile != 0 or args.batch == 1 or face_enhancer is not None or args.outscale != netscale:
            f = batch_buf.pop(0)
            try:
                if face_enhancer is not None:
                    # GFPGAN expects BGR; convert
                    f_bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                    _, _, out = face_enhancer.enhance(f_bgr, has_aligned=False, only_center_face=False, paste_back=True)
                else:
                    f_bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)  # <— add this
                    out, _ = upsampler.enhance(f_bgr, outscale=args.outscale)
            except RuntimeError as e:
                print('Error', e)
            else:
                writer.write_frame(out)
            pbar.update(1)

        # optional device sync to keep queue bounded
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
        except Exception:
            pass

    reader.close()
    writer.close()


def run(args):
    args.video_name = osp.splitext(os.path.basename(args.input))[0]
    video_save_path = osp.join(args.output, f'{args.video_name}_{args.suffix}.mp4')

    if args.extract_frame_first:
        tmp_frames_folder = osp.join(args.output, f'{args.video_name}_inp_tmp_frames')
        os.makedirs(tmp_frames_folder, exist_ok=True)
        os.system(f'ffmpeg -i "{args.input}" -qscale:v 1 -qmin 1 -qmax 1 -vsync 0 {tmp_frames_folder}/frame%08d.png')
        args.input = tmp_frames_folder

    num_gpus = torch.cuda.device_count()
    num_process = max(1, num_gpus * args.num_process_per_gpu)
    if num_process == 1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inference_video(args, video_save_path, device=device)
        return

    ctx = torch.multiprocessing.get_context('spawn')
    pool = ctx.Pool(num_process)
    os.makedirs(osp.join(args.output, f'{args.video_name}_out_tmp_videos'), exist_ok=True)
    pbar = tqdm(total=num_process, unit='sub_video', desc='inference')
    for i in range(num_process):
        sub_video_save_path = osp.join(args.output, f'{args.video_name}_out_tmp_videos', f'{i:03d}.mp4')
        pool.apply_async(
            inference_video,
            args=(args, sub_video_save_path, torch.device(i % num_gpus) if num_gpus else None, num_process, i),
            callback=lambda arg: pbar.update(1)
        )
    pool.close()
    pool.join()

    # concat sub videos
    vidlist = f'{args.output}/{args.video_name}_vidlist.txt'
    with open(vidlist, 'w') as f:
        for i in range(num_process):
            f.write(f'file \'{args.video_name}_out_tmp_videos/{i:03d}.mp4\'\n')

    cmd = [
        args.ffmpeg_bin, '-f', 'concat', '-safe', '0', '-i', vidlist,
        '-c', 'copy', f'{video_save_path}'
    ]
    print(' '.join(cmd))
    subprocess.call(cmd)
    shutil.rmtree(osp.join(args.output, f'{args.video_name}_out_tmp_videos'))
    if osp.exists(osp.join(args.output, f'{args.video_name}_inp_tmp_videos')):
        shutil.rmtree(osp.join(args.output, f'{args.video_name}_inp_tmp_videos'))
    os.remove(vidlist)


def main():
    """Real-ESRGAN video upscaler (AMD/ROCm-friendly; no VapourSynth)."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input video, image, or folder')
    parser.add_argument(
        '-n', '--model_name', type=str, default='realesr-animevideov3',
        help=('Model names: realesr-animevideov3 | RealESRGAN_x4plus_anime_6B | RealESRGAN_x4plus | '
              'RealESRNet_x4plus | RealESRGAN_x2plus | realesr-general-x4v3 | or any custom alias'))
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a custom weight (.pth/.safetensors); overrides -n lookup')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-dn', '--denoise_strength', type=float, default=0.5,
                        help='Only for realesr-general-x4v3 (0=keep noise, 1=strong denoise)')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='Final upsampling scale')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix for output filename')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size (0 = no tiling)')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding (8–16 typical)')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding per border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN for faces')
    parser.add_argument('--fp32', action='store_true', help='Use fp32 (default: fp16)')
    parser.add_argument('--fps', type=float, default=None, help='FPS of output video (default: input FPS)')
    parser.add_argument('--ffmpeg_bin', type=str, default='ffmpeg', help='Path to ffmpeg binary')
    parser.add_argument('--extract_frame_first', action='store_true')
    parser.add_argument('--num_process_per_gpu', type=int, default=1)
    parser.add_argument('--alpha_upsampler', type=str, default='realesrgan',
                        help='Alpha upsampler: realesrgan | bicubic')
    parser.add_argument('--ext', type=str, default='auto',
                        help='Image extension for image-folder mode')

    # NEW: performance knobs
    parser.add_argument('--encoder', type=str, default='libx264',
                        choices=['libx264', 'h264_vaapi', 'hevc_vaapi'],
                        help='Video encoder (VAAPI is fast on AMD)')
    parser.add_argument('--preset', type=str, default='faster',
                        help='x264 preset (ultrafast…placebo). Ignored for VAAPI.')
    parser.add_argument('--crf', type=int, default=18,
                        help='x264 CRF; larger = faster/smaller file. Ignored for VAAPI.')
    parser.add_argument('--qp', type=int, default=20,
                        help='VAAPI quality (global_quality). Lower = better quality, slower.')
    parser.add_argument('--batch', type=int, default=1,
                        help='True micro-batch when --tile 0 and no face enhance (try 2)')
    parser.add_argument('--queue_size', type=int, default=8,
                        help='Decode queue size (frames) for IO/compute overlap')
    parser.add_argument('--no_compile', action='store_true',
                        help='Disable torch.compile (Inductor) even if available')

    args = parser.parse_args()
    args.input = args.input.rstrip('/').rstrip('\\')
    os.makedirs(args.output, exist_ok=True)

    is_video = bool(mimetypes.guess_type(args.input)[0] and mimetypes.guess_type(args.input)[0].startswith('video'))
    if is_video and args.input.endswith('.flv'):
        mp4_path = args.input.replace('.flv', '.mp4')
        os.system(f'ffmpeg -i "{args.input}" -codec copy "{mp4_path}"')
        args.input = mp4_path

    if args.extract_frame_first and not is_video:
        args.extract_frame_first = False

    run(args)

    if args.extract_frame_first:
        tmp_frames_folder = osp.join(args.output, f'{args.video_name}_inp_tmp_frames')
        shutil.rmtree(tmp_frames_folder)


if __name__ == '__main__':
    main()
