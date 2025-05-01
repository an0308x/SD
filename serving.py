from flask import Flask, request, jsonify
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2

app = Flask(__name__)
pipeline = None

GLOBAL_SERVER = None

class Server:
    def __init__(self, approx_model_name, target_model_name) -> None:
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logging.info("begin load models")
        
        # Load approx model with config adjustment
        logging.info(f"Loading config for approx model: {approx_model_name}")
        try:
            approx_config = AutoConfig.from_pretrained(approx_model_name, trust_remote_code=True)
            if hasattr(approx_config, "rope_scaling") and isinstance(approx_config.rope_scaling, dict):
                logging.info(f"Original approx rope_scaling: {approx_config.rope_scaling}")
                required_keys = {'type', 'factor'}
                if set(approx_config.rope_scaling.keys()) != required_keys:
                    original_rope_config = approx_config.rope_scaling
                    rope_type = original_rope_config.get('rope_type') # Get type from 'rope_type' key
                    factor = original_rope_config.get('factor')
                    if rope_type is not None and factor is not None:
                         new_rope_scaling = {'type': rope_type, 'factor': factor}
                         logging.info(f"Modifying approx rope_scaling to: {new_rope_scaling}")
                         approx_config.rope_scaling = new_rope_scaling
                    else:
                         logging.warning(f"Could not find 'rope_type' or 'factor' in approx rope_scaling config: {original_rope_config}, attempting to load without modification.")
            
            logging.info("Loading approx model...")
            self._small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, config=approx_config, trust_remote_code=True).to(self._device)
            logging.info("Approx model loaded.")
        except Exception as e:
            logging.error(f"Failed to load approx model {approx_model_name}: {e}")
            raise

        # Load target model with config adjustment
        logging.info(f"Loading config for target model: {target_model_name}")
        try:
            target_config = AutoConfig.from_pretrained(target_model_name, trust_remote_code=True)
            if hasattr(target_config, "rope_scaling") and isinstance(target_config.rope_scaling, dict):
                logging.info(f"Original target rope_scaling: {target_config.rope_scaling}")
                required_keys = {'type', 'factor'}
                if set(target_config.rope_scaling.keys()) != required_keys:
                    original_rope_config = target_config.rope_scaling
                    rope_type = original_rope_config.get('rope_type') # Get type from 'rope_type' key
                    factor = original_rope_config.get('factor')
                    if rope_type is not None and factor is not None:
                        new_rope_scaling = {'type': rope_type, 'factor': factor}
                        logging.info(f"Modifying target rope_scaling to: {new_rope_scaling}")
                        target_config.rope_scaling = new_rope_scaling
                    else:
                        logging.warning(f"Could not find 'rope_type' or 'factor' in target rope_scaling config: {original_rope_config}, attempting to load without modification.")

            logging.info("Loading target model...")
            self._large_model = AutoModelForCausalLM.from_pretrained(target_model_name, config=target_config, trust_remote_code=True).to(self._device)
            logging.info("Target model loaded.")
        except Exception as e:
            logging.error(f"Failed to load target model {target_model_name}: {e}")
            raise

        # Load tokenizer
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(approx_model_name)
            logging.info("Tokenizer loaded.")
        except Exception as e:
            logging.error(f"Failed to load tokenizer for {approx_model_name}: {e}")
            raise
            
        logging.info("Finish load models") # Corrected typo
          
        self.num_tokens = 40
        self.top_k = 10
        self.top_p = 0.9
        
    def process_request(self, request : str) -> torch.Tensor:
        input_str = request['prompt']
        logging.info(f"recieve request {input_str}")
        input_ids = self._tokenizer.encode(input_str, return_tensors='pt').to(self._device)
        output = speculative_sampling(input_ids, 
                                      self._small_model, 
                                      self._large_model, self.num_tokens, 
                                      top_k = self.top_k, 
                                      top_p = self.top_p)
        generated_text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

# Set up a route to listen for inference requests
@app.route('/predict', methods=['POST'])
def predict():
    # Check the content type of the request
    if request.headers['Content-Type'] != 'application/json':
        return jsonify({'error': 'Invalid content type'})

    # Get the request data
    request_data = request.json

    # Perform inference
    result = GLOBAL_SERVER.process_request(request_data)

    # Return the inference results
    return jsonify(result)

if __name__ == '__main__':
    GLOBAL_SERVER = Server(approx_model_name="/share_nfs/fangjiarui/root/code/hf_models/bloom-560m",
           target_model_name="/share_nfs/fangjiarui/root/code/hf_models/bloomz-7b1")
    # Start the Flask service
    app.run(host='0.0.0.0', port=5000)
