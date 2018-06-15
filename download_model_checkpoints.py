import requests
import zipfile
import os
import shutil

target_ssd_model_dir = './model'
target_facenet_models_dir='./facenet/models'

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":

    ssd_model_path = os.path.join(os.getcwd(),'model')
    facenet_model_path = os.path.join(os.getcwd(),'facenet','models')
    temp_zip_directory = os.path.join(os.getcwd(),'tmp_zip_folder')
    src_ssd_path = os.path.join(temp_zip_directory,'tf-deep-facial-recognition-lite-models','ssd_model')

    print("Downloading .....")
    file_id = '1_YC_UNM6X95x8npDU2_plh-MI5gkb02x'
    destination = os.path.join(os.getcwd(),'tf-deep-facial-recognition-lite-models.zip')
    download_file_from_google_drive(file_id, destination)
    print("Zip file has been downloaded! ")
    print("Unzipping the files ....")
    if not os.path.exists(temp_zip_directory):
        os.makedirs(temp_zip_directory)
    if not os.path.exists(ssd_model_path):
        os.makedirs(ssd_model_path)

    with zipfile.ZipFile(os.path.join(os.getcwd(),"tf-deep-facial-recognition-lite-models.zip"),"r") as zip_ref:
        zip_ref.extractall(temp_zip_directory)

    src_ssd_path_files = os.listdir(src_ssd_path)

    for path_files in src_ssd_path_files:
        shutil.copy(os.path.join(src_ssd_path,path_files),ssd_model_path)

    shutil.copytree(os.path.join(temp_zip_directory,'tf-deep-facial-recognition-lite-models','facenet_models'),facenet_model_path)
    shutil.rmtree(temp_zip_directory)






