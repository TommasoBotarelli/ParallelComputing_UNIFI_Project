import os
import cv2
cv2.setNumThreads(1)
import numpy as np
import albumentations as A
from multiprocessing import Pool, cpu_count, Queue, Process, Manager
import time
import cProfile
import pstats
import math
import argparse
import copy

def create_random_transform():
    return A.Compose([
        A.OneOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.RandomRotate90(p=1)
        ], p=0.5),
        A.SomeOf([
            A.RandomBrightnessContrast(p=1),
            A.HueSaturationValue(p=1),
            A.CLAHE(p=1),
            A.GaussianBlur(p=1),
            A.Sharpen(p=1),
            A.RandomGamma(p=1),
            A.MotionBlur(p=1),
            A.ISONoise(p=1),
            A.ChannelShuffle(p=1)
        ], n=2, p=1),
        A.OneOf([
            A.ElasticTransform(p=1),
            A.GridDistortion(p=1),
            A.OpticalDistortion(p=1),
            A.Perspective(p=1)
        ], p=0.4),
    ])

###############################################################################################################################
###############################################################################################################################
# STANDARD PARALLEL IMPLEMENTATION

def augment_image(image, output_dir, name, ext, index, saving):  
    random_transform = create_random_transform() 
    augmented = random_transform(image=image)["image"]  # Apply random augmentation
    
    if saving:
        # Convert back to BGR for saving
        augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

        # Generate output filename (e.g., "aug_image1_1.jpg", "aug_image1_2.jpg", ...)
        output_filename = f"aug_{name}_{index}{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        cv2.imwrite(output_path, augmented)  # Save the augmented image


def process_images_parallel(input_dir, output_dir, n_augmentations=5, num_workers=None, saving=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files_raw = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]
    images_data = [(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), output_dir, os.path.splitext(os.path.basename(image_path))[0], os.path.splitext(os.path.basename(image_path))[1]) for image_path in image_files_raw]
    
    args = [(*data, str(index).zfill(6), saving) for data in images_data for index in range(n_augmentations)]
    
    # Use multiprocessing to process images in parallel
    with Pool(num_workers) as pool:
        pool.starmap(augment_image, args)


###############################################################################################################################
###############################################################################################################################
# STANDARD PARALLEL IMPLEMENTATION v2


def process_images_parallel_v2(input_dir, output_dir, n_augmentations=5, num_workers=None, saving=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files_raw = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]
    images_data = [(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), output_dir, os.path.splitext(os.path.basename(image_path))[0], os.path.splitext(os.path.basename(image_path))[1]) for image_path in image_files_raw]
    
    args = [(*copy.deepcopy(data), str(index).zfill(6), copy.deepcopy(saving)) for data in images_data for index in range(n_augmentations)]
    
    # Use multiprocessing to process images in parallel
    with Pool(num_workers) as pool:
        pool.starmap(augment_image, args)

###############################################################################################################################
###############################################################################################################################
# QUEUE PARALLEL IMPLEMENTATION 1

def save_images_from_queue(queue):
    while True:
        image_data = queue.get()
        if image_data is None:  # Sentinel value to stop the process
            break
        
        image, output_filename = image_data
        cv2.imwrite(output_filename, image)


def augment_image_queue(image, output_dir, name, ext, index, queue):  
    random_transform = create_random_transform() 
    augmented = random_transform(image=image)["image"]  # Apply random augmentation
    
    # Convert back to BGR for saving
    augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

    # Generate output filename (e.g., "aug_image1_1.jpg", "aug_image1_2.jpg", ...)
    output_filename = f"aug_{name}_{index}{ext}"
    output_path = os.path.join(output_dir, output_filename)
    
    queue.put((augmented, output_path))


def process_images_parallel_queue(input_dir, output_dir, n_augmentations=5, num_workers=None, saving=True):
    if not saving:
        print("Parallelization with queue not useful if saving files isn't allowed")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    manager = Manager()  # Use Manager to create a shared Queue
    queue = manager.Queue()

    image_files_raw = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]
    images_data = [(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), output_dir, os.path.splitext(os.path.basename(image_path))[0], os.path.splitext(os.path.basename(image_path))[1]) for image_path in image_files_raw]
    
    args = [(*data, str(index).zfill(6), queue) for data in images_data for index in range(n_augmentations)]

    saving_process = Process(target=save_images_from_queue, args=(queue, ))
    saving_process.start()
    
    # Use multiprocessing to process images in parallel
    with Pool(num_workers-1) as pool:
        pool.starmap(augment_image_queue, args)

    queue.put(None)
    saving_process.join()

###############################################################################################################################
###############################################################################################################################
# QUEUE PARALLEL IMPLEMENTATION sqrt

def process_images_parallel_queue_sqrt(input_dir, output_dir, n_augmentations=5, num_workers=None, saving=True):
    if not saving:
        print("Parallelization with queue not useful if saving files isn't allowed")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_savers = max(1, math.ceil(math.sqrt(num_workers)-1))
    #num_savers = max(1, math.ceil(num_workers/2))
    
    manager = Manager()  # Use Manager to create a shared Queue
    queues_raw = [manager.Queue() for _ in range(num_savers)]

    image_files_raw = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]
    images_data = [(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), output_dir, os.path.splitext(os.path.basename(image_path))[0], os.path.splitext(os.path.basename(image_path))[1]) for image_path in image_files_raw]
    
    args = [(*data, str(index).zfill(6), queues_raw[index%num_savers]) for data in images_data for index in range(n_augmentations)]

    saving_processes = [Process(target=save_images_from_queue, args=(queue, )) for queue in queues_raw]
    for saving_process in saving_processes:
        saving_process.start()
    
    # Use multiprocessing to process images in parallel
    with Pool(num_workers-num_savers) as pool:
        pool.starmap(augment_image_queue, args)

    for queue in queues_raw:
        queue.put(None)

    for saving_process in saving_processes:
        saving_process.join()

###############################################################################################################################
###############################################################################################################################
# READ ONE BY ONE PARALLEL IMPLEMENTATION

def augment_image_from_file(image_path, output_dir, index, saving):  
    random_transform = create_random_transform() 
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    augmented = random_transform(image=image)["image"]  # Apply random augmentation
    
    if saving:
        # Convert back to BGR for saving
        augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)

        # Generate output filename (e.g., "aug_image1_1.jpg", "aug_image1_2.jpg", ...)
        output_filename = f"aug_{name}_{index}{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        cv2.imwrite(output_path, augmented)  # Save the augmented image


def process_images_parallel_readFileOneByOne(input_dir, output_dir, n_augmentations=5, num_workers=None, saving=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]
    args = [(data, output_dir, str(index).zfill(6), saving) for data in image_files for index in range(n_augmentations)]
    
    # Use multiprocessing to process images in parallel
    with Pool(num_workers) as pool:
        pool.starmap(augment_image_from_file, args)


###############################################################################################################################
###############################################################################################################################
# SEQUENTIAL

def process_images_sequential(input_dir, output_dir, n_augmentations=5, saving=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]

    for image_path in image_files:
        image = cv2.imread(image_path)
        if image is None:
            return f"Failed to load {image_path}"
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        filename = os.path.basename(image_path)  # Extract filename (e.g., "image1.jpg")
        name, ext = os.path.splitext(filename)   # Split into name and extension
        for i in range(n_augmentations):
            index = str(i).zfill(6)
            augment_image(image, output_dir, name, ext, index, saving)

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
# MAIN

if __name__ == "__main__":
    output_directory_sequential = "images/output/sequential/"

    output_directory_parallel = "images/output/parallel/"
    output_directory_parallel_queue = "images/output/parallel_queue/"
    output_directory_parallel_1b1 = "images/output/parallel_1b1/"

    #'''

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Image Augmentation Script")
    parser.add_argument("--input_directory", type=str, required=True, help="Path to the input directory containing images")
    parser.add_argument("--n_augmentations", type=int, required=True, help="Number of augmentations per image")
    parser.add_argument("--num_workers", type=int, required=True, help="Number of workers for parallel processing")
    parser.add_argument("--sequential", action="store_true", help="Run the script in sequential computation mode")
    parser.add_argument("--parallel", action="store_true", help="Run the script in parallel computation mode")
    parser.add_argument("--saving", action="store_true", help="Save the files, execution times in this mode refers to computation time + saving time [no queue method run]")
    parser.add_argument("--num_iterations", type=int, default=1, help="Number of iterations to compute average execution time")
    args = parser.parse_args()

    input_directory = args.input_directory
    n_augmentations = args.n_augmentations
    num_workers = args.num_workers
    sequential = args.sequential
    parallel = args.parallel
    saving = args.saving
    num_iterations = args.num_iterations
    
    # Clean the output directories
    for directory in [output_directory_sequential, output_directory_parallel, output_directory_parallel_queue, output_directory_parallel_1b1]:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    print(f"Number of augmentations per image: {n_augmentations}")
    
    if sequential:
        print("-------------------------------------------------------------------------")
        total_duration = 0
        print(f"SEQUENTIAL processing...")
        for i in range(num_iterations):
            start_time = time.time()
            process_images_sequential(input_directory, output_directory_sequential, n_augmentations=n_augmentations, saving=saving)
            end_time = time.time()
            duration = end_time - start_time
            print(f"[{i}]: {duration:.2f} s")
            total_duration += duration
        print(f"Avg. execution time: {total_duration/num_iterations} s")
    
    if parallel:
        print("-------------------------------------------------------------------------")
        total_duration = 0
        print(f"PARALLEL ({num_workers} workers) processing...")
        for i in range(num_iterations):
            start_time = time.time()
            process_images_parallel(input_directory, output_directory_parallel, n_augmentations=n_augmentations, num_workers=num_workers, saving=saving)
            end_time = time.time()
            duration = end_time - start_time
            print(f"[{i}]: {duration:.2f} s")
            total_duration += duration
        print(f"Avg. execution time: {total_duration/num_iterations} s")

        print("-------------------------------------------------------------------------")
        total_duration = 0
        print(f"PARALLEL READ ONE BY ONE ({num_workers} workers) processing...")
        for i in range(num_iterations):
            start_time = time.time()
            process_images_parallel_readFileOneByOne(input_directory, output_directory_parallel, n_augmentations=n_augmentations, num_workers=num_workers, saving=saving)
            end_time = time.time()
            duration = end_time - start_time
            print(f"[{i}]: {duration:.2f} s")
            total_duration += duration
        print(f"Avg. execution time: {total_duration/num_iterations} s")

        if saving:
            print("-------------------------------------------------------------------------")
            print(f"PARALLEL QUEUE ({num_workers-1} workers, 1 saver) processing...")
            total_duration = 0
            for i in range(num_iterations):
                start_time = time.time()
                process_images_parallel_queue(input_directory, output_directory_parallel, n_augmentations=n_augmentations, num_workers=num_workers, saving=saving)
                end_time = time.time()
                duration = end_time - start_time
                print(f"[{i}]: {duration:.2f} s")
                total_duration += duration
            print(f"Avg. execution time: {total_duration/num_iterations} s")

            num_savers = max(1, math.ceil(math.sqrt(num_workers)) - 1)
            total_duration = 0
            print("-------------------------------------------------------------------------")
            print(f"PARALLEL QUEUE SQRT ({num_workers-num_savers} workers, {num_savers} savers) processing...")
            for i in range(num_iterations):
                start_time = time.time()
                process_images_parallel_queue_sqrt(input_directory, output_directory_parallel, n_augmentations=n_augmentations, num_workers=num_workers, saving=saving)
                end_time = time.time()
                duration = end_time - start_time
                print(f"[{i}]: {duration:.2f} s")
                total_duration += duration
            print(f"Avg. execution time: {total_duration/num_iterations} s")
    
    #'''
    '''
    cProfile.run('process_images_parallel("images/input_10", "images/output/parallel", 10, 32, True)', 'outputv1.prof')
    stats = pstats.Stats('outputv1.prof')
    stats.strip_dirs().sort_stats('time').print_stats(20)   
    '''
    '''
    cProfile.run('process_images_parallel("images/weak_scaling_inputs/2000_1000", "images/output/parallel", 100, 32, True)', 'outputv2.prof')
    stats = pstats.Stats('outputv2.prof')
    stats.strip_dirs().sort_stats('time').print_stats(20)   
    '''