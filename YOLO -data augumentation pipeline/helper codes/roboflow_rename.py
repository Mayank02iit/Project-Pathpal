import roboflow

rf = roboflow.Roboflow(api_key="m0M6g9jbLAAviaDoRFQ6")

# get a workspace
workspace = rf.workspace("pathpalws")

# Upload data set to a new/existing project
workspace.upload_dataset(
    "./test", # This is your dataset path
    "trial-fr7pr", # This will either create or get a dataset with the given ID
    num_workers=10,
    project_license="MIT",
    project_type="object-detection",
    batch_name=None,
    num_retries=0
)