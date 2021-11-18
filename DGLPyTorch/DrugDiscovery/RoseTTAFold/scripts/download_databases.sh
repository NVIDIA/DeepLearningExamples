# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

WEIGHTS_DIR="${1:-.}"
DATABASES_DIR="${2:-./databases}"

mkdir -p $DATABASES_DIR

echo "Downloading pre-trained model weights [1G]"
wget https://files.ipd.uw.edu/pub/RoseTTAFold/weights.tar.gz
tar xfz weights.tar.gz -C $WEIGHTS_DIR


# uniref30 [46G]
echo "Downloading UniRef30_2020_06 [46G]"
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz -P $DATABASES_DIR
mkdir -p $DATABASES_DIR/UniRef30_2020_06
tar xfz $DATABASES_DIR/UniRef30_2020_06_hhsuite.tar.gz -C $DATABASES_DIR/UniRef30_2020_06

# BFD [272G]
#wget https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz
#mkdir -p bfd
#tar xfz bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz -C ./bfd

# structure templates (including *_a3m.ffdata, *_a3m.ffindex) [over 100G]
echo "Downloading pdb100_2021Mar03 [over 100G]"
wget https://files.ipd.uw.edu/pub/RoseTTAFold/pdb100_2021Mar03.tar.gz -P $DATABASES_DIR
tar xfz $DATABASES_DIR/pdb100_2021Mar03.tar.gz -C $DATABASES_DIR/
