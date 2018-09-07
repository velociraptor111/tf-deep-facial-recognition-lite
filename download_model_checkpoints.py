import requests
import zipfile
import os
import shutil
import glob

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

    '''
        Target Directory
    '''
    root_model_path = os.path.join(os.getcwd(),'model')
    ssd_model_path = os.path.join(os.getcwd(),'model','ssd_models')
    posenet_model_path = os.path.join(os.getcwd(),'model','tf_pose_graph')
    facenet_model_path = os.path.join(os.getcwd(),'model','Facenet_Model')
    trained_svm_knn_model_directory = os.path.join(os.getcwd(),'trained_svm_knn_face_models')

    temp_zip_directory = os.path.join(os.getcwd(),'tmp_zip_folder')
    zip_file_destination = os.path.join(os.getcwd(), 'tf-deep-facial-recognition-lite-models.zip')

    '''
        Source Directory
    '''
    src_ssd_path = os.path.join(temp_zip_directory,'tf-deep-facial-recognition-lite-models','ssd_models')
    src_facenet_path = os.path.join(temp_zip_directory,'tf-deep-facial-recognition-lite-models','Facenet_Model')
    src_posenet_path = os.path.join(temp_zip_directory,'tf-deep-facial-recognition-lite-models','tf_pose_graph')

    print("Downloading ..... This will take awhile. Be patient.")
    file_id = '1tdNgX1qlq_nCtGuU7BKBUIQcL5zmsawW'
    download_file_from_google_drive(file_id, zip_file_destination)
    print("Zip file has been downloaded! ")
    print("Unzipping the files ....")
    '''
        Make necessary Directories 
    '''
    if not os.path.exists(temp_zip_directory):
        os.makedirs(temp_zip_directory)

    if not os.path.exists(trained_svm_knn_model_directory):
        os.makedirs(trained_svm_knn_model_directory)

    with zipfile.ZipFile(os.path.join(os.getcwd(),"tf-deep-facial-recognition-lite-models.zip"),"r") as zip_ref:
        zip_ref.extractall(temp_zip_directory)

    # Copy SSD Models
    src_ssd_path_files = os.listdir(src_ssd_path)
    shutil.copytree(src_ssd_path,ssd_model_path)

    # Copy Posenet Models
    src_posenet_path_files = os.listdir(src_posenet_path)
    shutil.copytree(src_posenet_path,posenet_model_path)

    # Copy Facenet Models
    src_facenet_path_files = os.listdir(src_facenet_path)
    shutil.copytree(src_facenet_path,facenet_model_path)

    # Copy the svm and knn models into their own repository
    pkl_files = glob.glob(src_facenet_path+'/*.pkl')
    for pkl_file_path in pkl_files:
        shutil.copy(pkl_file_path,trained_svm_knn_model_directory)

    # Delete the temp unzipped folder
    shutil.rmtree(temp_zip_directory)
    # Delete the actual zip file
    os.remove(zip_file_destination)






