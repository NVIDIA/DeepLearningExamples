# FastPitch: Voice Modification with Transformations of Pitch

This readme details how to run the Jupyter notebook for FastPitch inference using different pitch transformations.

## Build and run the FastPitch Docker container

1. Clone the repository:
   ```
   git clone https://github.com/NVIDIA/DeepLearningExamples.git
   cd DeepLearningExamples/PyTorch/SpeechSynthesis/FastPitch
   ```

2. Build the container for the FastPitch model:
   ```bash
   docker build . -t fastpitch:latest
   ```

3. Launch the container of FastPitch model. By default port `8888` if forwarded.
   ```bash
   bash scripts/docker/interactive.sh
   ```
## Run Jupyter notebook

Inside the container, navigate to the `notebooks/` directory and start the Jupyter notebook server:
```
cd notebooks
jupyter notebook --ip='*' --port=8888 --allow-root
```
Then navigate a web browser to the IP address or hostname of the host machine at port `8888`:
```
http://[host machine]:8888
```
Use the token listed in the output from running the jupyter command to log in, for example:
```
http://[host machine]:8888/?token=aae96ae9387cd28151868fee318c3b3581a2d794f3b25c6b
```
