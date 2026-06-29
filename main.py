from heads.vqa.inference import run_inference
from heads.detr.inference import run_inference as run_detr_inference

# TODO: Only run the Dinov3 Backbone once and pass the features to the heads
# TODO: Also compute the PCA for interpretability


answers = run_inference(
    image_path="image.jpg",
    question="What disease is affecting my maize plant?",
)


detection_results = run_detr_inference(
    image_path="image.jpg",
)


print(answers)
print(detection_results)