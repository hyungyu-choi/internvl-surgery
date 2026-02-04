"""
Example usage of InternVL3 surgical phase recognition
"""
import os
from models import InternVL3Wrapper
from utils import preprocess_image, predict_phase_from_embedding
import pickle


def example_single_image_inference():
    """Example: Infer phase for a single image"""
    
    # Initialize model
    model = InternVL3Wrapper(
        model_path="OpenGVLab/InternVL3-8B",
        device='cuda',
        layer_idx=13
    )
    
    # Load phase embeddings
    with open("path/to/phase_embeddings.pkl", 'rb') as f:
        embedding_data = pickle.load(f)
    phase_stats = embedding_data['phase_stats']
    
    # Process single image
    image_path = "path/to/frame.jpg"
    pixel_values, num_patches_list = preprocess_image(image_path)
    
    # Extract embeddings
    frame_embedding = model.extract_visual_embeddings(pixel_values, num_patches_list)
    
    # Predict phase
    predicted_phase, distances, sorted_distances = predict_phase_from_embedding(
        frame_embedding, phase_stats
    )
    
    print(f"Predicted phase: {predicted_phase}")
    print(f"Distance: {distances[predicted_phase]:.4f}")
    print(f"\nTop 3 candidates:")
    for phase, dist in sorted_distances[:3]:
        print(f"  {phase}: {dist:.4f}")


def example_dataset_usage():
    """Example: Using dataset classes"""
    from data import Cholec80Dataset
    from torch.utils.data import DataLoader
    
    # Create dataset
    dataset = Cholec80Dataset(
        frames_dir="path/to/video/frames",
        input_size=448,
        max_num_patches=12
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Phase names: {dataset.get_phase_names()}")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Iterate
    for batch in dataloader:
        pixel_values = batch['pixel_values']
        labels = batch['label']
        print(f"Batch shape: {pixel_values.shape}")
        break


def example_custom_transform():
    """Example: Using custom transform"""
    from data import Cholec80Dataset
    from utils import preprocess_image
    
    def my_custom_transform(image_path):
        # Your custom preprocessing here
        return preprocess_image(image_path, input_size=224)
    
    dataset = Cholec80Dataset(
        frames_dir="path/to/frames",
        transform=my_custom_transform
    )


if __name__ == "__main__":
    print("InternVL3 Surgical Phase Recognition - Examples")
    print("="*60)
    
    # Uncomment to run examples
    # example_single_image_inference()
    # example_dataset_usage()
    # example_custom_transform()
    
    print("\nSee source code for example implementations")
