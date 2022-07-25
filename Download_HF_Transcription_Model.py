# from transformers.pipelines import pipeline
from transformers import AutoProcessor, AutoModelForCTC

model_id = "loulely/XLSR_300M_Fine_Tuning_FR_2"
auth_token = "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"

model = AutoModelForCTC.from_pretrained(model_id, use_auth_token=auth_token)
processor = AutoProcessor.from_pretrained(model_id, use_auth_token=auth_token)


model.save_pretrained('./wav2vec2_fine_tuned_fr')
processor.save_pretrained('./wav2vec2_fine_tuned_fr')
