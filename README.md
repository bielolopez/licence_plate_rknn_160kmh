PARA REALIZAR MULTIPLES LINEAS UTILICE DICHAS LINEAS QUE LAS GENERALICE PARA MODELOS U OTROS SISTEMAS. FUNCIONO HASTA EN UN SISTEMA DE 72 NPUs.  y camaras de 120 mil fotograms por segundos y de resolucion satelitarl.


"""""""""""""""""""""""""
import threading
from rknn.api import rknn_lite

# Global variables (or passed as arguments to threads)
RKNN_MODEL_PATH = 'your_model.rknn'
NUM_THREADS = 4

def inference_thread_func(thread_id, input_data_queue, output_results_queue):
    # Initialize RKNNLite within the thread
    rknn_lite_obj = rknn_lite.RKNNLite()
    print(f"Thread {thread_id}: Loading RKNN model...")
    ret = rknn_lite_obj.load_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        print(f"Thread {thread_id}: Load RKNN model failed!")
        return

    while True:
        input_data = input_data_queue.get()
        if input_data is None:  # Sentinel to stop the thread
            break

        # Pre-process input_data (e.g., resize, normalize)
        # ...

        # Perform inference
        outputs = rknn_lite_obj.inference(inputs=[input_data])

        # Post-process outputs if needed
        # ...

        output_results_queue.put((thread_id, outputs))

    rknn_lite_obj.release()
    print(f"Thread {thread_id}: RKNNLite released.")

# Main execution
if __name__ == '__main__':
    input_queue = Queue()
    output_queue = Queue()
    threads = []

    for i in range(NUM_THREADS):
        thread = threading.Thread(target=inference_thread_func, args=(i, input_queue, output_queue))
        threads.append(thread)
        thread.start()

    # Populate input_queue with data (e.g., image frames)
    # ...

    # Add sentinels to stop threads
    for _ in range(NUM_THREADS):
        input_queue.put(None)

    # Wait for threads to finish and collect results
    for thread in threads:
        thread.join()

    # Process collected results
    # ...
""""""""""""""
