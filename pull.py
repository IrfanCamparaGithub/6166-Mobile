from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="icampara/spirkappmobile",
    repo_type="space",                   
    local_dir=r"C:\Users\Irfan\Downloads\mobile",
    local_dir_use_symlinks=False,        
    resume_download=True                 
)