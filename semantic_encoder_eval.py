# SCUIA semantic encoder evaluation method: Measuring the cosine similarity between the model's features of test image and
# text prompt features.

from torch.utils.data import DataLoader
from dataloader import *
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm
from PIL import Image
import clip


def _prepare_clip_image_tensor(x):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    return x


def build_semantic_inference_context(model):
    text_prompts = clip.tokenize(
        ["a perfect underwater photo.", "a fair underwater photo.", "a bad underwater photo."]
    ).to("cuda")
    text_features = model.clip_model.encode_text(text_prompts).unsqueeze(0).detach()
    normalizer = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
    )
    return {
        'local_encoder': model.image_encoder,
        'text_features': text_features,
        'normalizer': normalizer,
    }


def compute_semantic_scores(model, test_dataset, img_dir, data_loc, sample_callback=None):
     
    text_prompt = clip.tokenize(["a perfect underwater photo.", "a fair underwater photo.", "a bad underwater photo."]).to("cuda")
    local_encoder = model.image_encoder
    text_features = model.clip_model.encode_text(text_prompt).unsqueeze(0).detach()  
    normalizer = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)) # Using the same dataloader for image and semantic model. Hence, as only semantic model requires normalization, we do it here.

    with torch.no_grad():
        # print("Computing the scores for the semantic model")
        names = []
        moss = []
        scores = []



        dataset = TestDataset(img_dir, data_loc)
        loader = DataLoader(dataset, batch_size= 1, shuffle=False)
        
        for batch, (img, mos, img_name) in enumerate(tqdm(loader)):
            input = normalizer(_prepare_clip_image_tensor(img))
            image_features = local_encoder(input.to("cuda")).unsqueeze(1)
            score = F.cosine_similarity(image_features, text_features, dim=-1)
            difference = 10.0 * (score[:, 1] - score[:, 0])
            scaled_score = 1 / (1 + torch.exp(difference))

            if scaled_score.shape == torch.Size([]):
                sample_scores = [scaled_score.item()]
            else:
                sample_scores = scaled_score.tolist()

            sample_mos = mos.tolist()
            sample_names = list(img_name)

            scores.extend(sample_scores)
            moss.extend(sample_mos)
            names.extend(sample_names)

            if sample_callback is not None:
                for idx, file_name in enumerate(sample_names):
                    sample_callback(file_name, sample_mos[idx], sample_scores[idx])

    return names, scores, moss


def compute_semantic_score_single_image(model, test_image_path, inference_context=None):
    if inference_context is None:
        inference_context = build_semantic_inference_context(model)
    local_encoder = inference_context['local_encoder']
    text_features = inference_context['text_features']
    normalizer = inference_context['normalizer']

    with torch.no_grad():
        # print("Computing the SCUIA semantic encoder score for a single image")
        
        scores = []

        x = Image.open(test_image_path)
        transform = transforms.ToTensor()
        x = transform(x)
        if x.shape[0] <3:
            x = torch.cat([x]*3, dim=0)
        x = normalizer(_prepare_clip_image_tensor(x))

        image_features = local_encoder(x.to("cuda")).unsqueeze(1)
        score = F.cosine_similarity(image_features, text_features, dim=-1)
        difference = 10.0 * (score[:, 1] - score[:, 0])
        scaled_score = 1 / (1 + torch.exp(difference))

        if scaled_score.shape == torch.Size([]):
            scores.append(scaled_score.item())
        else:
            scores.extend(scaled_score.tolist())

    return scores
