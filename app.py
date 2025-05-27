import gradio as gr
from clean import DateSorted
import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
from huggingface_hub import hf_hub_download
import os
import tempfile

# Huggingface dataset info
runs = [
    {
        "name": "DOLMA-1k",
        "repo_id": "HuggingFaceTB/gutenburg-longattn-1000",
        "inference_file": "dolma-1k/output.jsonl",
        "sample_file": "dolma-1k/sample_output.jsonl"
    },
    {
        "name": "Gutenberg-1k",
        "repo_id": "HuggingFaceTB/gutenburg-longattn-1000",
        "inference_file": "output.jsonl",
        "sample_file": "sample_output.jsonl"
    },
    {
        "name": "Gutenberg-10k",
        "repo_id": "HuggingFaceTB/gutenburg-longattn-1000",
        "inference_file": "gutenburg-10k/output.jsonl",
        "sample_file": "gutenburg-10k/sample_output.jsonl"
    },
    {
        "name": "Local",
        "repo_id": "Local",
        "inference_file": "/fsx/jason/LongAttn/output.jsonl",
        "sample_file": "/fsx/jason/LongAttn/sample_output.jsonl"
    }
]

# Global variables to store current data
current_data = {
    "ds": None,
    "tds": None,
    "scores": None,
    "sorted_scores": None,
    "proportion_means": None,
    "variance_means": None,
    "token_distance_scores": None,
    "file_path": None
}

def load_data_for_run(run_name):
    """Load data for the selected run"""
    # Find the run configuration
    run_config = None
    for run in runs:
        if run["name"] == run_name:
            run_config = run
            break
    
    if not run_config:
        raise ValueError(f"Run {run_name} not found")
    
    # Create a temporary directory for downloaded files
    temp_dir = tempfile.mkdtemp()
    
    if run_config["repo_id"] == "Local":
        # Use local files
        inference_path = run_config["inference_file"]
        file_path = run_config["sample_file"]
    else:
        # Download from Huggingface Hub
        inference_path = hf_hub_download(
            repo_id=run_config["repo_id"], 
            filename=run_config["inference_file"], 
            repo_type="dataset", 
            token=os.getenv('HF_TOKEN')
        )
        file_path = hf_hub_download(
            repo_id=run_config["repo_id"], 
            filename=run_config["sample_file"], 
            repo_type="dataset", 
            token=os.getenv('HF_TOKEN')
        )
    
    output_path = os.path.join(temp_dir, "tmp.tmp")
    
    # Create DateSorted instance and process data
    ds = DateSorted(
        inference_path=inference_path,
        file_path=file_path,
        output_path=output_path
    )
    
    tds, scores = ds.sorted_file()
    
    # Extract the three score types
    proportion_means = [item['proportion_mean'] for item in scores]
    variance_means = [item['variance_mean'] for item in scores]
    token_distance_scores = [item['token_distance_score'] for item in scores]
    
    # Sort scores by token_distance_score for slider indexing
    sorted_scores = sorted(scores, key=lambda x: x['token_distance_score'])
    
    # Update global data
    current_data.update({
        "ds": ds,
        "tds": tds,
        "scores": scores,
        "sorted_scores": sorted_scores,
        "proportion_means": proportion_means,
        "variance_means": variance_means,
        "token_distance_scores": token_distance_scores,
        "file_path": file_path
    })
    
    return len(scores)

def create_plots(selected_index=0):
    if not current_data["sorted_scores"]:
        return None
        
    # Get selected document's scores
    selected_token_score = current_data["sorted_scores"][selected_index]['token_distance_score']
    selected_proportion = current_data["sorted_scores"][selected_index]['proportion_mean']
    selected_variance = current_data["sorted_scores"][selected_index]['variance_mean']
    
    # Create figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Proportion Mean
    ax1.hist(current_data["proportion_means"], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Distribution of Proportion Mean', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Proportion Mean')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    # Add vertical line for selected document
    ax1.axvline(x=selected_proportion, color='red', linestyle='--', linewidth=2,
                label=f'Selected: {selected_proportion:.4f}')
    ax1.legend(loc='upper right')

    # Plot 2: Variance Mean
    ax2.hist(current_data["variance_means"], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_title('Distribution of Variance Mean', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Variance Mean')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    # Format x-axis to show scientific notation clearly
    ax2.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    # Add vertical line for selected document
    ax2.axvline(x=selected_variance, color='red', linestyle='--', linewidth=2,
                label=f'Selected: {selected_variance:.2e}')
    ax2.legend(loc='upper right')

    # Plot 3: Token Distance Score
    ax3.hist(current_data["token_distance_scores"], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.set_title('Distribution of Token Distance Score', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Token Distance Score')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # Add vertical line for selected document on token distance score plot
    ax3.axvline(x=selected_token_score, color='red', linestyle='--', linewidth=2, 
                label=f'Selected: {selected_token_score:.4f}')
    ax3.legend(loc='upper right')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    
    # Convert BytesIO to PIL Image
    img = Image.open(buf)
    plt.close(fig)
    
    return img

def get_stats():
    if not current_data["proportion_means"]:
        return "No data loaded"
        
    return f"""
    Basic Statistics:
    Proportion Mean - Min: {min(current_data["proportion_means"]):.4f}, Max: {max(current_data["proportion_means"]):.4f}, Mean: {np.mean(current_data["proportion_means"]):.4f}
    Variance Mean - Min: {min(current_data["variance_means"]):.2e}, Max: {max(current_data["variance_means"]):.2e}, Mean: {np.mean(current_data["variance_means"]):.2e}
    Token Distance Score - Min: {min(current_data["token_distance_scores"]):.4f}, Max: {max(current_data["token_distance_scores"]):.4f}, Mean: {np.mean(current_data["token_distance_scores"]):.4f}
    """

def get_document_by_index(index):
    if not current_data["sorted_scores"] or not current_data["file_path"]:
        return "No data loaded", "No data loaded"
        
    # Get the data_id of the selected document
    data_id = current_data["sorted_scores"][index]['data_id']
    
    # Find the document in the file
    with jsonlines.open(current_data["file_path"], 'r') as reader:
        for data in reader:
            if data["data_id"] == data_id:
                doc = data
                break
    
    # Prepare display information
    score_info = f"""
    Document ID: {data_id}
    Token Distance Score: {current_data["sorted_scores"][index]['token_distance_score']:.4f}
    Proportion Mean: {current_data["sorted_scores"][index]['proportion_mean']:.4f}
    Variance Mean: {current_data["sorted_scores"][index]['variance_mean']:.2e}
    """
    
    return score_info, doc['content']

def update_display(index):
    # Get document info
    score_info, content = get_document_by_index(index)
    # Update plot with new vertical line position
    plot = create_plots(index)
    return score_info, content, plot

def change_run(run_name):
    """Handle run selection change"""
    try:
        # Load data for the new run
        max_docs = load_data_for_run(run_name)
        
        # Reset to first document
        score_info, content = get_document_by_index(0)
        plot = create_plots(0)
        stats = get_stats()
        
        # Return updated values and new slider maximum
        return (
            gr.Slider(minimum=0, maximum=max_docs-1, step=1, value=0, 
                     label="Document Selection (sorted by Token Distance Score)"),
            score_info,
            content,
            plot,
            stats
        )
    except Exception as e:
        error_msg = f"Error loading run {run_name}: {str(e)}"
        return (
            gr.Slider(minimum=0, maximum=0, step=1, value=0, 
                     label="Document Selection (sorted by Token Distance Score)"),
            error_msg,
            error_msg,
            None,
            error_msg
        )

# Initialize with first run
initial_run = runs[0]["name"]
max_docs = load_data_for_run(initial_run)

# Create Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# LongAttn Doc Viewer")
    gr.Markdown("This is a sample of documents from the Gutenberg dataset that are chunked and scored using the [LongAttn](https://github.com/Lyun0912-wu/LongAttn) method.")
    
    with gr.Row():
        run_dropdown = gr.Dropdown(
            choices=[run["name"] for run in runs],
            value=initial_run,
            label="Select Dataset"
        )
    
    with gr.Row():
        plot_output = gr.Image(value=create_plots(0), label="Score Distributions")
    
    with gr.Row():
        stats_output = gr.Textbox(value=get_stats(), label="Statistics", lines=4)
    
    with gr.Row():
        slider = gr.Slider(
            minimum=0, 
            maximum=max_docs-1, 
            step=1, 
            value=0, 
            label="Document Selection (sorted by Token Distance Score)"
        )
    
    with gr.Row():
        score_info = gr.Textbox(label="Document Score Information", lines=5)
        document_content = gr.Textbox(label="Document Content", lines=15)
    
    # Initialize with first document
    score_info_val, content_val = get_document_by_index(0)
    score_info.value = score_info_val
    document_content.value = content_val
    
    # Connect dropdown to change_run function
    run_dropdown.change(
        fn=change_run, 
        inputs=run_dropdown, 
        outputs=[slider, score_info, document_content, plot_output, stats_output]
    )
    
    # Connect slider to update function
    slider.change(fn=update_display, inputs=slider, outputs=[score_info, document_content, plot_output])

# Launch the app
app.launch(share=True)
