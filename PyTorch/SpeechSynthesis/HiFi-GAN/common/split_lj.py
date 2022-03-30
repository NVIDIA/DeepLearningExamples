# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from csv import QUOTE_NONE
from pathlib import Path

import pandas as pd


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-d', '--metadata-path', type=str,
                        default='./metadata.csv',
                        help='Path to LJSpeech dataset metadata')
    parser.add_argument('--filelists-path', default='data/filelists', type=str,
                        help='Directory to generate filelists to')
    parser.add_argument('--log-file', type=str, default='split_log.json',
                         help='Filename for logging')
    parser.add_argument('--subsets', type=str, nargs='+',
                        choices=['all', 'train', 'val', 'test'],
                        default=['all', 'train', 'val'],
                        help='Subsets to generate')
    parser.add_argument('--add-transcript', action='store_true',
                        help='Add text columns to filelists')
    parser.add_argument('--add-pitch', action='store_true',
                        help='Add pitch columns to filelists')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Increase verbosity')
    parser.add_argument('--dry-run', action='store_true',
                        help='Do not create actual files')
    return parser


def generate_filelist(subset, meta, base_fname, cols_to_dump, args):
    subset_suffix = f'_{subset}' if subset != 'all' else ''
    fpath = Path(args.filelists_path) / f'{base_fname}{subset_suffix}.txt'

    if subset == 'all':
        subset_meta = meta[meta.index.map(
            lambda fname: fname not in discard_train_ids_v3)]
    elif subset == 'val':
        subset_meta = meta[meta.index.map(lambda fname: fname in val_ids)]
    elif subset == 'test':
        subset_meta = meta[meta.index.map(lambda fname: fname in test_ids)]
    elif subset == 'train':
        subset_meta = meta[meta.index.map(
            lambda fname: (fname not in val_ids and fname not in test_ids
                           and fname not in discard_train_ids_v3)
        )]
    else:
        raise ValueError(f'Unknown subset: {subset}')

    print(f'Writing {len(subset_meta)} rows to {fpath}')
    if args.verbose:
        print(subset_meta.reset_index()[cols_to_dump].head())
    if not args.dry_run:
        fpath.parent.mkdir(parents=True, exist_ok=True)
        subset_meta.to_csv(fpath, sep='|', header=None, quoting=QUOTE_NONE,
                           index=None, columns=cols_to_dump)


def main():
    parser = argparse.ArgumentParser(
        description='LJSpeech filelists generation')
    parser = parse_args(parser)
    args, unk_args = parser.parse_known_args()
    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')

    meta = pd.read_csv(args.metadata_path, sep='|', index_col='fname',
                       names=['fname', 'txt', 'norm_txt'], quoting=QUOTE_NONE)
    meta['wav_fname'] = meta.index.map(lambda stem: f'wavs/{stem}.wav')
    meta['pitch_fname'] = meta.index.map(lambda stem: f'pitch/{stem}.pt')

    cols_to_dump = ['wav_fname']
    filelist_base_name = 'ljs_audio'
    if args.add_pitch:
        cols_to_dump.append('pitch_fname')
        filelist_base_name += '_pitch'
    if args.add_transcript:
        cols_to_dump.append('norm_txt')
        filelist_base_name += '_text'

    # Fix incrrect transcripts
    if args.add_transcript:
        for lj_id, txt in corrected_transcripts.items():
            print('Trying to correct', lj_id)
            meta.loc[meta['wav_fname'].str.contains(lj_id), 'norm_txt'] = txt

    for subset in args.subsets:
        generate_filelist(subset, meta, filelist_base_name, cols_to_dump, args)


corrected_transcripts = {
    'LJ031-0175': "O'Donnell tried to persuade Mrs. Kennedy to leave the area, but she refused. She said that she intended to stay with her husband.",
    'LJ034-0138': "they saw and heard Brennan describing what he had seen. Norman stated, quote,",
}

# ASR-recognized French words that could hinder generalization
discard_train_ids_v3 = {
    'LJ011-0058', 'LJ012-0205', 'LJ016-0257', 'LJ018-0396',
}

val_ids = {
    'LJ001-0110', 'LJ002-0018', 'LJ002-0043', 'LJ003-0111', 'LJ003-0345',
    'LJ004-0045', 'LJ004-0096', 'LJ004-0152', 'LJ005-0014', 'LJ005-0079',
    'LJ005-0201', 'LJ007-0154', 'LJ008-0111', 'LJ008-0258', 'LJ008-0278',
    'LJ008-0294', 'LJ008-0307', 'LJ009-0076', 'LJ009-0114', 'LJ009-0238',
    'LJ011-0096', 'LJ012-0035', 'LJ012-0042', 'LJ012-0161', 'LJ012-0235',
    'LJ012-0250', 'LJ013-0164', 'LJ014-0010', 'LJ014-0020', 'LJ014-0030',
    'LJ014-0076', 'LJ014-0110', 'LJ014-0263', 'LJ015-0203', 'LJ016-0020',
    'LJ016-0138', 'LJ016-0179', 'LJ016-0288', 'LJ016-0318', 'LJ017-0044',
    'LJ017-0070', 'LJ017-0131', 'LJ018-0081', 'LJ018-0098', 'LJ018-0239',
    'LJ019-0186', 'LJ019-0257', 'LJ019-0273', 'LJ021-0066', 'LJ021-0145',
    'LJ022-0023', 'LJ024-0083', 'LJ026-0068', 'LJ027-0052', 'LJ027-0141',
    'LJ028-0008', 'LJ028-0081', 'LJ028-0093', 'LJ028-0275', 'LJ028-0307',
    'LJ028-0335', 'LJ028-0506', 'LJ029-0022', 'LJ029-0032', 'LJ031-0038',
    'LJ031-0070', 'LJ031-0134', 'LJ031-0202', 'LJ033-0047', 'LJ034-0042',
    'LJ034-0053', 'LJ034-0160', 'LJ034-0198', 'LJ035-0019', 'LJ035-0129',
    'LJ036-0077', 'LJ036-0103', 'LJ036-0174', 'LJ037-0234', 'LJ038-0199',
    'LJ039-0075', 'LJ039-0223', 'LJ040-0002', 'LJ040-0027', 'LJ042-0096',
    'LJ042-0129', 'LJ043-0002', 'LJ043-0030', 'LJ045-0140', 'LJ045-0230',
    'LJ046-0058', 'LJ046-0146', 'LJ046-0184', 'LJ047-0044', 'LJ047-0148',
    'LJ048-0194', 'LJ048-0228', 'LJ049-0026', 'LJ049-0050', 'LJ050-0168'
}

test_ids = {
    'LJ001-0015', 'LJ001-0051', 'LJ001-0063', 'LJ001-0072', 'LJ001-0079',
    'LJ001-0094', 'LJ001-0096', 'LJ001-0102', 'LJ001-0153', 'LJ001-0173',
    'LJ001-0186', 'LJ002-0096', 'LJ002-0105', 'LJ002-0106', 'LJ002-0171',
    'LJ002-0174', 'LJ002-0220', 'LJ002-0225', 'LJ002-0253', 'LJ002-0260',
    'LJ002-0261', 'LJ002-0289', 'LJ002-0298', 'LJ002-0299', 'LJ003-0011',
    'LJ003-0088', 'LJ003-0107', 'LJ003-0140', 'LJ003-0159', 'LJ003-0211',
    'LJ003-0230', 'LJ003-0282', 'LJ003-0299', 'LJ003-0319', 'LJ003-0324',
    'LJ004-0009', 'LJ004-0024', 'LJ004-0036', 'LJ004-0077', 'LJ004-0083',
    'LJ004-0239', 'LJ005-0019', 'LJ005-0072', 'LJ005-0086', 'LJ005-0099',
    'LJ005-0248', 'LJ005-0252', 'LJ005-0253', 'LJ005-0257', 'LJ005-0264',
    'LJ005-0265', 'LJ005-0294', 'LJ006-0021', 'LJ006-0040', 'LJ006-0043',
    'LJ006-0044', 'LJ006-0082', 'LJ006-0084', 'LJ006-0088', 'LJ006-0137',
    'LJ006-0149', 'LJ006-0202', 'LJ006-0268', 'LJ007-0071', 'LJ007-0075',
    'LJ007-0076', 'LJ007-0085', 'LJ007-0090', 'LJ007-0112', 'LJ007-0125',
    'LJ007-0130', 'LJ007-0150', 'LJ007-0158', 'LJ007-0170', 'LJ007-0233',
    'LJ008-0054', 'LJ008-0085', 'LJ008-0098', 'LJ008-0121', 'LJ008-0181',
    'LJ008-0182', 'LJ008-0206', 'LJ008-0215', 'LJ008-0228', 'LJ008-0266',
    'LJ009-0037', 'LJ009-0041', 'LJ009-0061', 'LJ009-0074', 'LJ009-0084',
    'LJ009-0106', 'LJ009-0124', 'LJ009-0126', 'LJ009-0172', 'LJ009-0184',
    'LJ009-0192', 'LJ009-0194', 'LJ009-0276', 'LJ009-0280', 'LJ009-0286',
    'LJ010-0027', 'LJ010-0030', 'LJ010-0038', 'LJ010-0062', 'LJ010-0065',
    'LJ010-0083', 'LJ010-0157', 'LJ010-0158', 'LJ010-0219', 'LJ010-0228',
    'LJ010-0257', 'LJ010-0281', 'LJ010-0297', 'LJ011-0041', 'LJ011-0048',
    'LJ011-0118', 'LJ011-0141', 'LJ011-0245', 'LJ012-0015', 'LJ012-0021',
    'LJ012-0049', 'LJ012-0054', 'LJ012-0067', 'LJ012-0188', 'LJ012-0189',
    'LJ012-0194', 'LJ012-0219', 'LJ012-0230', 'LJ012-0257', 'LJ012-0271',
    'LJ013-0005', 'LJ013-0045', 'LJ013-0055', 'LJ013-0091', 'LJ013-0098',
    'LJ013-0104', 'LJ013-0109', 'LJ013-0213', 'LJ014-0029', 'LJ014-0094',
    'LJ014-0121', 'LJ014-0128', 'LJ014-0142', 'LJ014-0146', 'LJ014-0171',
    'LJ014-0186', 'LJ014-0194', 'LJ014-0199', 'LJ014-0224', 'LJ014-0233',
    'LJ014-0265', 'LJ014-0306', 'LJ014-0326', 'LJ015-0001', 'LJ015-0005',
    'LJ015-0007', 'LJ015-0025', 'LJ015-0027', 'LJ015-0036', 'LJ015-0043',
    'LJ015-0052', 'LJ015-0144', 'LJ015-0194', 'LJ015-0218', 'LJ015-0231',
    'LJ015-0266', 'LJ015-0289', 'LJ015-0308', 'LJ016-0007', 'LJ016-0049',
    'LJ016-0054', 'LJ016-0077', 'LJ016-0089', 'LJ016-0117', 'LJ016-0125',
    'LJ016-0137', 'LJ016-0192', 'LJ016-0205', 'LJ016-0233', 'LJ016-0238',
    'LJ016-0241', 'LJ016-0248', 'LJ016-0264', 'LJ016-0274', 'LJ016-0277',
    'LJ016-0283', 'LJ016-0314', 'LJ016-0347', 'LJ016-0367', 'LJ016-0380',
    'LJ016-0417', 'LJ016-0426', 'LJ017-0035', 'LJ017-0050', 'LJ017-0059',
    'LJ017-0102', 'LJ017-0108', 'LJ017-0133', 'LJ017-0134', 'LJ017-0164',
    'LJ017-0183', 'LJ017-0189', 'LJ017-0190', 'LJ017-0226', 'LJ017-0231',
    'LJ018-0031', 'LJ018-0129', 'LJ018-0130', 'LJ018-0159', 'LJ018-0206',
    'LJ018-0211', 'LJ018-0215', 'LJ018-0218', 'LJ018-0231', 'LJ018-0244',
    'LJ018-0262', 'LJ018-0276', 'LJ018-0278', 'LJ018-0288', 'LJ018-0309',
    'LJ018-0349', 'LJ018-0354', 'LJ019-0042', 'LJ019-0052', 'LJ019-0055',
    'LJ019-0129', 'LJ019-0145', 'LJ019-0161', 'LJ019-0169', 'LJ019-0179',
    'LJ019-0180', 'LJ019-0201', 'LJ019-0202', 'LJ019-0221', 'LJ019-0241',
    'LJ019-0248', 'LJ019-0270', 'LJ019-0289', 'LJ019-0317', 'LJ019-0318',
    'LJ019-0335', 'LJ019-0344', 'LJ019-0348', 'LJ019-0355', 'LJ019-0368',
    'LJ019-0371', 'LJ020-0085', 'LJ020-0092', 'LJ020-0093', 'LJ021-0012',
    'LJ021-0025', 'LJ021-0026', 'LJ021-0040', 'LJ021-0078', 'LJ021-0091',
    'LJ021-0110', 'LJ021-0115', 'LJ021-0139', 'LJ021-0140', 'LJ023-0016',
    'LJ023-0033', 'LJ023-0047', 'LJ023-0056', 'LJ023-0089', 'LJ023-0122',
    'LJ024-0018', 'LJ024-0019', 'LJ024-0034', 'LJ024-0054', 'LJ024-0102',
    'LJ025-0081', 'LJ025-0098', 'LJ025-0118', 'LJ025-0129', 'LJ025-0157',
    'LJ026-0034', 'LJ026-0052', 'LJ026-0054', 'LJ026-0108', 'LJ026-0140',
    'LJ026-0148', 'LJ027-0006', 'LJ027-0176', 'LJ027-0178', 'LJ028-0023',
    'LJ028-0136', 'LJ028-0138', 'LJ028-0145', 'LJ028-0168', 'LJ028-0212',
    'LJ028-0226', 'LJ028-0278', 'LJ028-0289', 'LJ028-0340', 'LJ028-0349',
    'LJ028-0357', 'LJ028-0410', 'LJ028-0416', 'LJ028-0421', 'LJ028-0459',
    'LJ028-0462', 'LJ028-0494', 'LJ028-0502', 'LJ029-0004', 'LJ029-0052',
    'LJ029-0060', 'LJ029-0096', 'LJ029-0114', 'LJ029-0197', 'LJ030-0006',
    'LJ030-0014', 'LJ030-0021', 'LJ030-0032', 'LJ030-0035', 'LJ030-0063',
    'LJ030-0084', 'LJ030-0105', 'LJ030-0125', 'LJ030-0162', 'LJ030-0196',
    'LJ030-0197', 'LJ030-0238', 'LJ031-0008', 'LJ031-0014', 'LJ031-0041',
    'LJ031-0058', 'LJ031-0109', 'LJ031-0122', 'LJ031-0165', 'LJ031-0185',
    'LJ031-0189', 'LJ032-0012', 'LJ032-0025', 'LJ032-0027', 'LJ032-0045',
    'LJ032-0085', 'LJ032-0103', 'LJ032-0164', 'LJ032-0180', 'LJ032-0204',
    'LJ032-0206', 'LJ032-0261', 'LJ033-0042', 'LJ033-0055', 'LJ033-0056',
    'LJ033-0072', 'LJ033-0093', 'LJ033-0120', 'LJ033-0152', 'LJ033-0159',
    'LJ033-0174', 'LJ033-0183', 'LJ033-0205', 'LJ034-0035', 'LJ034-0041',
    'LJ034-0072', 'LJ034-0097', 'LJ034-0117', 'LJ034-0123', 'LJ034-0134',
    'LJ034-0166', 'LJ034-0197', 'LJ035-0014', 'LJ035-0082', 'LJ035-0155',
    'LJ035-0164', 'LJ036-0067', 'LJ036-0104', 'LJ036-0169', 'LJ037-0001',
    'LJ037-0002', 'LJ037-0007', 'LJ037-0053', 'LJ037-0061', 'LJ037-0081',
    'LJ037-0208', 'LJ037-0248', 'LJ037-0249', 'LJ037-0252', 'LJ038-0035',
    'LJ038-0047', 'LJ038-0264', 'LJ039-0027', 'LJ039-0059', 'LJ039-0076',
    'LJ039-0088', 'LJ039-0096', 'LJ039-0118', 'LJ039-0125', 'LJ039-0139',
    'LJ039-0148', 'LJ039-0154', 'LJ039-0192', 'LJ039-0207', 'LJ039-0227',
    'LJ040-0018', 'LJ040-0052', 'LJ040-0097', 'LJ040-0110', 'LJ040-0176',
    'LJ040-0201', 'LJ041-0022', 'LJ041-0070', 'LJ041-0195', 'LJ041-0199',
    'LJ042-0097', 'LJ042-0130', 'LJ042-0133', 'LJ042-0135', 'LJ042-0194',
    'LJ042-0198', 'LJ042-0219', 'LJ042-0221', 'LJ042-0230', 'LJ043-0010',
    'LJ043-0016', 'LJ043-0047', 'LJ043-0107', 'LJ043-0140', 'LJ043-0188',
    'LJ044-0004', 'LJ044-0013', 'LJ044-0047', 'LJ044-0105', 'LJ044-0125',
    'LJ044-0135', 'LJ044-0137', 'LJ044-0139', 'LJ044-0158', 'LJ044-0224',
    'LJ044-0237', 'LJ045-0015', 'LJ045-0033', 'LJ045-0045', 'LJ045-0082',
    'LJ045-0090', 'LJ045-0092', 'LJ045-0096', 'LJ045-0177', 'LJ045-0178',
    'LJ045-0190', 'LJ045-0194', 'LJ045-0216', 'LJ045-0228', 'LJ045-0234',
    'LJ046-0016', 'LJ046-0033', 'LJ046-0055', 'LJ046-0092', 'LJ046-0105',
    'LJ046-0111', 'LJ046-0113', 'LJ046-0179', 'LJ046-0191', 'LJ046-0226',
    'LJ047-0015', 'LJ047-0022', 'LJ047-0049', 'LJ047-0056', 'LJ047-0073',
    'LJ047-0075', 'LJ047-0093', 'LJ047-0097', 'LJ047-0126', 'LJ047-0158',
    'LJ047-0197', 'LJ047-0202', 'LJ047-0240', 'LJ048-0033', 'LJ048-0053',
    'LJ048-0069', 'LJ048-0112', 'LJ048-0143', 'LJ048-0197', 'LJ048-0200',
    'LJ048-0222', 'LJ048-0252', 'LJ048-0288', 'LJ048-0289', 'LJ049-0022',
    'LJ049-0115', 'LJ049-0154', 'LJ049-0196', 'LJ049-0202', 'LJ050-0004',
    'LJ050-0022', 'LJ050-0029', 'LJ050-0031', 'LJ050-0056', 'LJ050-0069',
    'LJ050-0084', 'LJ050-0090', 'LJ050-0118', 'LJ050-0137', 'LJ050-0161',
    'LJ050-0162', 'LJ050-0188', 'LJ050-0209', 'LJ050-0223', 'LJ050-0235'
}


if __name__ == '__main__':
    main()
