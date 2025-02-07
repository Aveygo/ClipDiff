# Idea

## The issue
We have identified a general trend of SOTA models requiring richer textual embeddings to produce high quality images. These textual embeddings typically come from CLIP (Contrastive Language-Image Pre-training) models, which is a method created by OpenAI to pair images to their captions, which is why they excel in conditioning image generation tasks. However, for CLIP to work, it needs a high performing image classifier such as ViT to encode input images into its corresponding embeddings. Such classifier backbones tend to be quite large and difficult to scale due to their use of transformers, which presents itself as an opportunity for improvement.

## Our solution
We propose a novel zero shot image classifier inspired by Differential Transformers which denoises intermediate latent maps during training/ inferencing within the text domain. We hypothesize that by translating this technique into the vision domain, we can achieve similar levels of efficiency that were found in the paper, most notably a 40% reduction in parameter count while maintaining the same performance. The feasibility of this work is heavily reliant on ConvNext, which has similarly shown the successful use of translating transformer based solutions into the convolutional domain.

## Impact
Ultimately, we hope to create higher quality embeddings that can be used in downstream tasks such as for image generation, and for other such zero-shot tasks such as vector databases and image searching. Such a discovery can also help refine the benefits / drawbacks of transformers in the vision domain which has recently seen more scrutiny due to the influence of ConvNext. 
