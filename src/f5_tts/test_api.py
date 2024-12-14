import random
import sys
from importlib.resources import files

import soundfile as sf
import tqdm
from cached_path import cached_path

from f5_tts.infer.utils_infer import (
    hop_length,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    transcribe,
    target_sample_rate,
)
from f5_tts.model import DiT, UNetT
from f5_tts.model.utils import seed_everything


class F5TTS:
    def __init__(
        self,
        model_type="F5-TTS",
        ckpt_file="",
        vocab_file="",
        ode_method="euler",
        use_ema=True,
        vocoder_name="vocos",
        local_path=None,
        device=None,
        hf_cache_dir=None,
    ):
        # Initialize parameters
        self.final_wave = None
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.seed = -1
        self.mel_spec_type = vocoder_name

        # Set device
        if device is not None:
            self.device = device
        else:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        # Load models
        self.load_vocoder_model(vocoder_name, local_path=local_path, hf_cache_dir=hf_cache_dir)
        self.load_ema_model(
            model_type, ckpt_file, vocoder_name, vocab_file, ode_method, use_ema, hf_cache_dir=hf_cache_dir
        )

    def load_vocoder_model(self, vocoder_name, local_path=None, hf_cache_dir=None):
        self.vocoder = load_vocoder(vocoder_name, local_path is not None, local_path, self.device, hf_cache_dir)

    def load_ema_model(self, model_type, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema, hf_cache_dir=None):
        if model_type == "F5-TTS":
            if not ckpt_file:
                if mel_spec_type == "vocos":
                    ckpt_file = str(
                        cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors", cache_dir=hf_cache_dir)
                    )
                elif mel_spec_type == "bigvgan":
                    ckpt_file = str(
                        cached_path("hf://SWivid/F5-TTS/F5TTS_Base_bigvgan/model_1250000.pt", cache_dir=hf_cache_dir)
                    )
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            model_cls = DiT
        elif model_type == "E2-TTS":
            if not ckpt_file:
                ckpt_file = str(
                    cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors", cache_dir=hf_cache_dir)
                )
            model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
            model_cls = UNetT
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.ema_model = load_model(
            model_cls, model_cfg, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema, self.device
        )

    def transcribe(self, ref_audio, language=None):
        return transcribe(ref_audio, language)

    def export_wav(self, wav, file_wave, remove_silence=False):
        sf.write(file_wave, wav, self.target_sample_rate)

        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def export_spectrogram(self, spect, file_spect):
        save_spectrogram(spect, file_spect)

    def infer(
        self,
        ref_file,
        ref_text,
        gen_text,
        show_info=print,
        progress=tqdm,
        target_rms=0.1,
        cross_fade_duration=0.15,
        sway_sampling_coef=-1,
        cfg_strength=2,
        nfe_step=32,
        speed=1.0,
        fix_duration=None,
        remove_silence=False,
        file_wave=None,
        file_spect=None,
        seed=-1,
    ):
        if seed == -1:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        self.seed = seed

        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text, device=self.device)

        wav, sr, spect = infer_process(
            ref_file,
            ref_text,
            gen_text,
            self.ema_model,
            self.vocoder,
            self.mel_spec_type,
            show_info=show_info,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=self.device,
        )

        if file_wave is not None:
            self.export_wav(wav, file_wave, remove_silence)

        if file_spect is not None:
            self.export_spectrogram(spect, file_spect)

        return wav, sr, spect


if __name__ == "__main__":
    f5tts = F5TTS(
        ckpt_file="/home/snail/F5-TTS/ckpts/F5TTS_Base/model_1200000.pt",
        vocab_file="/home/snail/F5-TTS/ckpts/F5TTS_Base/vocab.txt",
        local_path="/home/snail/F5-TTS/ckpts/vocos-mel-24khz",
    )

    wav, sr, spect = f5tts.infer(
        ref_file=str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")),
        ref_text="some call me nature, others call me mother nature.",
        # gen_text="""I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring. Respect me and I'll nurture you; ignore me and you shall face the consequences.""",
        # ref_file=str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")),
        # ref_text="",
        gen_text="""《红楼梦》：中国古典文学的巅峰之作。《红楼梦》，又名《石头记》，是中国古典文学的巅峰之作，由清代作家曹雪芹创作，后由高鹗续写完成。全书以贾、史、王、薛四大家族的兴衰为背景，通过对贾宝玉、林黛玉、薛宝钗等众多人物的细腻描绘，展现了封建社会末期的家族兴衰、社会变迁以及人性的复杂与多样。小说以贾宝玉与林黛玉、薛宝钗的爱情悲剧为主线，融合了丰富的诗词、园林、饮食、服饰等细节，展现了作者深厚的文化素养和对生活的敏锐观察力。通过对大观园的描写，读者仿佛置身于那个华丽而脆弱的封建世界，感受到其中的繁华与暗流。《红楼梦》不仅是一部爱情小说，更是一部反映社会现实的巨著。书中通过对人物命运的刻画，揭示了封建制度的腐朽与人性的挣扎。尤其是对女性角色的塑造，如林黛玉的才情与脆弱，薛宝钗的端庄与智慧，王熙凤的精明与强势，展现了不同性格的女性在封建社会中的生存状态和心理活动。此外，小说还蕴含着丰富的哲理和深刻的思想。曹雪芹通过对人生无常、命运多舛的描绘，表达了对世事的感慨和对理想生活的向往。《红楼梦》的艺术成就极高，无论是人物塑造、情节设计，还是语言运用，都达到了炉火纯青的境地，被誉为“百年难得一书”。《红楼梦》的影响深远，不仅在中国文学史上占据重要地位，还对后世的文学创作产生了深刻影响。它被翻译成多种语言，传播到世界各地，成为了解中国古代文化的重要窗口。同时，关于《红楼梦》的研究也形成了一门独立的学科——红学，吸引了无数学者投身其中，探讨其各种深层次的意义和价值。总之，《红楼梦》以其丰富的内容、深刻的思想和高超的艺术手法，成为中华文化的瑰宝。它不仅是一部文学巨著，更是一面反映人性与社会的镜子，值得每一位读者细细品味，深入思考。""",
        file_wave=str(files("f5_tts").joinpath("../../tests/api_out.wav")),
        file_spect=str(files("f5_tts").joinpath("../../tests/api_out.png")),
        seed=-1,  # random seed = -1
    )

    print("seed :", f5tts.seed)
