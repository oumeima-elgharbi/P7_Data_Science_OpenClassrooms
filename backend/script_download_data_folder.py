import os
from gcloud import storage

# The credentials are saved as environment variables

# if deploy == True :
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "google-credentials.json"


def create_folder_from_blob_name(blob_name):
    """
    Creates folders locally if they don't exist
    example 'data/data_train_application.csv.gz'
    :param blob_name: (string)
    :return: None
    :rtype: None

    :UC: any
    """
    blob_name_list = blob_name.split('/')
    # we create nested folders in locally
    for i in range(1, len(blob_name_list)):
        new_folder = '/'.join(blob_name_list[:i])
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)


def download_blob_as_file(project_name, bucket_name, prefix, folder):
    """

    :param project_name: (string)
    :param bucket_name: (string) 'your-bucket-name'
    :param prefix: (string) 'your-bucket-directory/'
    :param folder: (string) 'your-local-directory/'
    :return:
    """
    # Connect to Google Storage and get bucket
    storage_client = storage.Client(project=project_name)
    bucket = storage_client.bucket(bucket_name)

    # Retrieve all blobs with a prefix matching the folder
    blobs = list(bucket.list_blobs(prefix=prefix))

    for blob in blobs:
        filename = blob.name
        # we create the nested local folders according to the blob name
        create_folder_from_blob_name(filename)
        if not filename.endswith("/"):
            blob.download_to_filename(folder + filename)


if __name__ == "__main__":
    print("Starting download")  # project-7-oc-376509
    download_blob_as_file(project_name="project-7-oc-376509",
                          bucket_name="p7-data",
                          prefix="resources",  # name of blob where we have stored the data to download
                          folder="")  # path_root + "/"
    print("End of download")
