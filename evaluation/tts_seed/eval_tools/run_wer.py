import os
import string
import sys

import scipy
import soundfile as sf
import zhconv
from funasr import AutoModel
from jiwer import process_words
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from zhon.hanzi import punctuation


punctuation_all = punctuation + string.punctuation

wav_res_text_path = sys.argv[1]
res_path = sys.argv[2]
lang = sys.argv[3]  # zh or en
device = os.environ.get("WER_DEVICE", "cuda:0")


def load_en_model():
    model_id = os.environ.get("WHISPER_MODEL", "openai/whisper-large-v3")
    cache_dir = os.environ.get("WHISPER_CACHE_DIR")
    processor = WhisperProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir).to(device)
    return processor, model


def load_zh_model():
    paraformer_model = os.environ.get("PARAFORMER_MODEL", "/path/to/paraformer-zh")
    # device 由 WER_DEVICE 控制（cpu / cuda:0），无显卡时用 cpu
    return AutoModel(model=paraformer_model, device=device, disable_update=True)


def process_one(hypo, truth):
    raw_truth = truth
    raw_hypo = hypo

    for x in punctuation_all:
        if x == "'":
            continue
        truth = truth.replace(x, "")
        hypo = hypo.replace(x, "")

    truth = truth.replace("  ", " ")
    hypo = hypo.replace("  ", " ")

    if lang == "zh":
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo])
    elif lang == "en":
        truth = truth.lower()
        hypo = hypo.lower()
    else:
        raise NotImplementedError

    measures = process_words(truth, hypo)
    ref_list = truth.split(" ")
    wer = measures.wer
    subs = measures.substitutions / len(ref_list)
    dele = measures.deletions / len(ref_list)
    inse = measures.insertions / len(ref_list)
    return (raw_truth, raw_hypo, wer, subs, dele, inse)


def run_asr(wav_res_text_path, res_path):
    if lang == "en":
        processor, model = load_en_model()
    elif lang == "zh":
        model = load_zh_model()
    else:
        raise NotImplementedError

    params = []
    for line in open(wav_res_text_path).readlines():
        line = line.strip()
        if len(line.split("|")) == 2:
            wav_res_path, text_ref = line.split("|")
        elif len(line.split("|")) == 3:
            wav_res_path, wav_ref_path, text_ref = line.split("|")
        elif len(line.split("|")) == 4:
            wav_res_path, _, text_ref, wav_ref_path = line.split("|")
        else:
            continue

        if not os.path.exists(wav_res_path):
            continue
        params.append((wav_res_path, text_ref))

    with open(res_path, "w") as fout:
        for wav_res_path, text_ref in tqdm(params):
            if lang == "en":
                wav, sr = sf.read(wav_res_path)
                if sr != 16000:
                    wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
                input_features = processor(wav, sampling_rate=16000, return_tensors="pt").input_features
                input_features = input_features.to(device)
                forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
                predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            else:
                res = model.generate(input=wav_res_path, batch_size_s=300)
                transcription = zhconv.convert(res[0]["text"], "zh-cn")

            raw_truth, raw_hypo, wer, subs, dele, inse = process_one(transcription, text_ref)
            fout.write(f"{wav_res_path}\t{wer}\t{raw_truth}\t{raw_hypo}\t{inse}\t{dele}\t{subs}\n")
            fout.flush()


run_asr(wav_res_text_path, res_path)
