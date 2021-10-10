# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import py3Dmol


def execute_pipeline(sequence):
    if os.path.isfile(sequence):
        with open(sequence, "r") as f:
            title = f.readlines()[0][2:]
        print(f"Running inference on {title}")
        os.system(f"bash run_inference_pipeline.sh {sequence}")
    else:
        try:
            with open("temp_input.fa", "w") as f:
                f.write(f"> {sequence[:8]}...\n")
                f.write(sequence.upper())
            print(f"Running inference on {sequence[:8]}...")
            os.system(f"bash run_inference_pipeline.sh temp_input.fa")
        except:
            print("Unable to run the pipeline.")
            raise


def display_pdb(path_to_pdb):
    with open(path_to_pdb) as ifile:
        protein = "".join([x for x in ifile])
    view = py3Dmol.view(width=400, height=300)
    view.addModelsAsFrames(protein)
    view.setStyle({'model': -1}, {"cartoon": {'color': 'spectrum'}})
    view.zoomTo()
    view.show()


def cleanup():
    os.system("rm t000*")
