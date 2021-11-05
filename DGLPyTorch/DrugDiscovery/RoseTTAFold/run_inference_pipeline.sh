#!/bin/bash
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# make the script stop when error (non-true exit code) is occurred
set -e
CPU="32"  # number of CPUs to use
MEM="64" # max memory (in GB)

# Inputs:
IN="$1"                # input.fasta
#WDIR=`realpath -s $2`  # working folder
DATABASES_DIR="${2:-/databases}"
WEIGHTS_DIR="${3:-/weights}"
WDIR="${4:-.}"



LEN=`tail -n1 $IN | wc -m`

mkdir -p $WDIR/logs

###########################################################
# 1. generate MSAs
############################################################
if [ ! -s $WDIR/t000_.msa0.a3m ]
then
    echo "Running HHblits - looking for the MSAs"
    /workspace/rf/input_prep/make_msa.sh $IN $WDIR $CPU $MEM $DATABASES_DIR > $WDIR/logs/make_msa.stdout 2> $WDIR/logs/make_msa.stderr
fi


############################################################
# 2. predict secondary structure for HHsearch run
############################################################
if [ ! -s $WDIR/t000_.ss2 ]
then
    echo "Running PSIPRED - looking for the Secondary Structures"
    /workspace/rf/input_prep/make_ss.sh $WDIR/t000_.msa0.a3m $WDIR/t000_.ss2 > $WDIR/logs/make_ss.stdout 2> $WDIR/logs/make_ss.stderr
fi


############################################################
# 3. search for templates
############################################################
if [ ! -s $WDIR/t000_.hhr ]
then
    echo "Running hhsearch - looking for the Templates"
    /workspace/rf/input_prep/prepare_templates.sh $WDIR $CPU $MEM $DATABASES_DIR > $WDIR/logs/prepare_templates.stdout 2> $WDIR/logs/prepare_templates.stderr
fi


############################################################
# 4. end-to-end prediction
############################################################
DB="$DATABASES_DIR/pdb100_2021Mar03/pdb100_2021Mar03"
if [ ! -s $WDIR/t000_.3track.npz ]
then
    echo "Running end-to-end prediction"
    python /workspace/rf/network/predict_e2e.py \
      -m $WEIGHTS_DIR \
      -i $WDIR/t000_.msa0.a3m \
      -o /results/output.e2e \
      --hhr $WDIR/t000_.hhr \
      --atab $WDIR/t000_.atab \
      --db $DB
fi
echo "Done."
echo "Output saved as /results/output.e2e.pdb"

# 1> $WDIR/log/network.stdout 2> $WDIR/log/network.stderr