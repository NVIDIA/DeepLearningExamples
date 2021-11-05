import sys,os,json
import tempfile
import numpy as np

from arguments import *
from utils import *
from pyrosetta import *
from pyrosetta.rosetta.protocols.minimization_packing import MinMover

vdw_weight = {0: 3.0, 1: 5.0, 2: 10.0}
rsr_dist_weight = {0: 3.0, 1: 2.0, 3: 1.0}
rsr_orient_weight = {0: 1.0, 1: 1.0, 3: 0.5}

def main():

    ########################################################
    # process inputs
    ########################################################

    # read params
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    with open(scriptdir + '/data/params.json') as jsonfile:
        params = json.load(jsonfile)

    # get command line arguments
    args = get_args(params)
    print(args)
    if os.path.exists(args.OUT):
        return

    # init PyRosetta
    init_cmd = list()
    init_cmd.append("-multithreading:interaction_graph_threads 1 -multithreading:total_threads 1")
    init_cmd.append("-hb_cen_soft")
    init_cmd.append("-detect_disulf -detect_disulf_tolerance 2.0") # detect disulfide bonds based on Cb-Cb distance (CEN mode) or SG-SG distance (FA mode)
    init_cmd.append("-relax:dualspace true -relax::minimize_bond_angles -default_max_cycles 200")
    init_cmd.append("-mute all")
    init(" ".join(init_cmd))

    # read and process restraints & sequence
    seq = read_fasta(args.FASTA)
    L = len(seq)
    params['seq'] = seq
    rst = gen_rst(params)

    ########################################################
    # Scoring functions and movers
    ########################################################
    sf = ScoreFunction()
    sf.add_weights_from_file(scriptdir + '/data/scorefxn.wts')

    sf1 = ScoreFunction()
    sf1.add_weights_from_file(scriptdir + '/data/scorefxn1.wts')

    sf_vdw = ScoreFunction()
    sf_vdw.add_weights_from_file(scriptdir + '/data/scorefxn_vdw.wts')

    sf_cart = ScoreFunction()
    sf_cart.add_weights_from_file(scriptdir + '/data/scorefxn_cart.wts')

    mmap = MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(False)
    mmap.set_jump(True)

    min_mover1 = MinMover(mmap, sf1, 'lbfgs_armijo_nonmonotone', 0.001, True)
    min_mover1.max_iter(1000)

    min_mover_vdw = MinMover(mmap, sf_vdw, 'lbfgs_armijo_nonmonotone', 0.001, True)
    min_mover_vdw.max_iter(500)

    min_mover_cart = MinMover(mmap, sf_cart, 'lbfgs_armijo_nonmonotone', 0.000001, True)
    min_mover_cart.max_iter(300)
    min_mover_cart.cartesian(True)


    if not os.path.exists("%s_before_relax.pdb"%('.'.join(args.OUT.split('.')[:-1]))): 
        ########################################################
        # initialize pose
        ########################################################
        pose0 = pose_from_sequence(seq, 'centroid')

        # mutate GLY to ALA
        for i,a in enumerate(seq):
            if a == 'G':
                mutator = rosetta.protocols.simple_moves.MutateResidue(i+1,'ALA')
                mutator.apply(pose0)
                print('mutation: G%dA'%(i+1))

        if (args.bb == ''):
            print('setting random (phi,psi,omega)...')
            set_random_dihedral(pose0)
        else:
            print('setting predicted (phi,psi,omega)...')
            bb = np.load(args.bb)
            set_predicted_dihedral(pose0,bb['phi'],bb['psi'],bb['omega'])

        remove_clash(sf_vdw, min_mover_vdw, pose0)

        Emin = 99999.9

        ########################################################
        # minimization
        ########################################################

        for run in range(params['NRUNS']):
            # define repeat_mover here!! (update vdw weights: weak (1.0) -> strong (10.0)
            sf.set_weight(rosetta.core.scoring.vdw, vdw_weight.setdefault(run, 10.0))
            sf.set_weight(rosetta.core.scoring.atom_pair_constraint, rsr_dist_weight.setdefault(run, 1.0))
            sf.set_weight(rosetta.core.scoring.dihedral_constraint, rsr_orient_weight.setdefault(run, 0.5))
            sf.set_weight(rosetta.core.scoring.angle_constraint, rsr_orient_weight.setdefault(run, 0.5))
            
            min_mover = MinMover(mmap, sf, 'lbfgs_armijo_nonmonotone', 0.001, True)
            min_mover.max_iter(1000)

            repeat_mover = RepeatMover(min_mover, 3)

            #
            pose = Pose()
            pose.assign(pose0)
            pose.remove_constraints()

            if run > 0:

                # diversify backbone
                dphi = np.random.uniform(-10,10,L)
                dpsi = np.random.uniform(-10,10,L)
                for i in range(1,L+1):
                    pose.set_phi(i,pose.phi(i)+dphi[i-1])
                    pose.set_psi(i,pose.psi(i)+dpsi[i-1])

                # remove clashes
                remove_clash(sf_vdw, min_mover_vdw, pose)
            
            # Save checkpoint
            if args.save_chk:
                pose.dump_pdb("%s_run%d_init.pdb"%('.'.join(args.OUT.split('.')[:-1]), run))

            if args.mode == 0:

                # short
                print('short')
                add_rst(pose, rst, 3, 12, params)
                repeat_mover.apply(pose)
                remove_clash(sf_vdw, min_mover1, pose)
                min_mover_cart.apply(pose)
                if args.save_chk:
                    pose.dump_pdb("%s_run%d_mode%d_step%d.pdb"%('.'.join(args.OUT.split('.')[:-1]), run, args.mode, 0))

                # medium
                print('medium')
                add_rst(pose, rst, 12, 24, params)
                repeat_mover.apply(pose)
                remove_clash(sf_vdw, min_mover1, pose)
                min_mover_cart.apply(pose)
                if args.save_chk:
                    pose.dump_pdb("%s_run%d_mode%d_step%d.pdb"%('.'.join(args.OUT.split('.')[:-1]), run, args.mode, 1))

                # long
                print('long')
                add_rst(pose, rst, 24, len(seq), params)
                repeat_mover.apply(pose)
                remove_clash(sf_vdw, min_mover1, pose)
                min_mover_cart.apply(pose)
                if args.save_chk:
                    pose.dump_pdb("%s_run%d_mode%d_step%d.pdb"%('.'.join(args.OUT.split('.')[:-1]), run, args.mode, 2))

            elif args.mode == 1:

                # short + medium
                print('short + medium')
                add_rst(pose, rst, 3, 24, params)
                repeat_mover.apply(pose)
                remove_clash(sf_vdw, min_mover1, pose)
                min_mover_cart.apply(pose)
                if args.save_chk:
                    pose.dump_pdb("%s_run%d_mode%d_step%d.pdb"%('.'.join(args.OUT.split('.')[:-1]), run, args.mode, 0))

                # long
                print('long')
                add_rst(pose, rst, 24, len(seq), params)
                repeat_mover.apply(pose)
                remove_clash(sf_vdw, min_mover1, pose)
                min_mover_cart.apply(pose)
                if args.save_chk:
                    pose.dump_pdb("%s_run%d_mode%d_step%d.pdb"%('.'.join(args.OUT.split('.')[:-1]), run, args.mode, 1))

            elif args.mode == 2:

                # short + medium + long
                print('short + medium + long')
                add_rst(pose, rst, 3, len(seq), params)
                repeat_mover.apply(pose)
                remove_clash(sf_vdw, min_mover1, pose)
                min_mover_cart.apply(pose)
                if args.save_chk:
                    pose.dump_pdb("%s_run%d_mode%d_step%d.pdb"%('.'.join(args.OUT.split('.')[:-1]), run, args.mode, 0))

            # check whether energy has decreased
            pose.conformation().detect_disulfides() # detect disulfide bonds
            E = sf_cart(pose)
            if E < Emin:
                print("Energy(iter=%d): %.1f --> %.1f (accept)"%(run, Emin, E))
                Emin = E
                pose0.assign(pose)
            else:
                print("Energy(iter=%d): %.1f --> %.1f (reject)"%(run, Emin, E))

        # mutate ALA back to GLY
        for i,a in enumerate(seq):
            if a == 'G':
                mutator = rosetta.protocols.simple_moves.MutateResidue(i+1,'GLY')
                mutator.apply(pose0)
                print('mutation: A%dG'%(i+1))

        ########################################################
        # fix backbone geometry
        ########################################################
        pose0.remove_constraints()

        # apply more strict criteria to detect disulfide bond
        # Set options for disulfide tolerance -> 1.0A
        print (rosetta.basic.options.get_real_option('in:detect_disulf_tolerance'))
        rosetta.basic.options.set_real_option('in:detect_disulf_tolerance', 1.0)
        print (rosetta.basic.options.get_real_option('in:detect_disulf_tolerance'))
        pose0.conformation().detect_disulfides()

        # Converto to all atom representation
        switch = SwitchResidueTypeSetMover("fa_standard")
        switch.apply(pose0)

        # idealize problematic local regions if exists
        idealize = rosetta.protocols.idealize.IdealizeMover()
        poslist = rosetta.utility.vector1_unsigned_long()

        scorefxn=create_score_function('empty')
        scorefxn.set_weight(rosetta.core.scoring.cart_bonded, 1.0)
        scorefxn.score(pose0)

        emap = pose0.energies()
        print("idealize...")
        for res in range(1,L+1):
            cart = emap.residue_total_energy(res)
            if cart > 50:
                poslist.append(res)
                print( "idealize %d %8.3f"%(res,cart) )

        if len(poslist) > 0:
            idealize.set_pos_list(poslist)
            try:
                idealize.apply(pose0)

            except:
                print('!!! idealization failed !!!')

        # Save checkpoint
        if args.save_chk:
            pose0.dump_pdb("%s_before_relax.pdb"%'.'.join(args.OUT.split('.')[:-1]))

    else: # checkpoint exists
        pose0 = pose_from_file("%s_before_relax.pdb"%('.'.join(args.OUT.split('.')[:-1])))
        #
        print ("to centroid")
        switch_cen = SwitchResidueTypeSetMover("centroid")
        switch_cen.apply(pose0)
        #
        print ("detect disulfide bonds")
        # Set options for disulfide tolerance -> 1.0A
        print (rosetta.basic.options.get_real_option('in:detect_disulf_tolerance'))
        rosetta.basic.options.set_real_option('in:detect_disulf_tolerance', 1.0)
        print (rosetta.basic.options.get_real_option('in:detect_disulf_tolerance'))
        pose0.conformation().detect_disulfides()
        #
        print ("to all atom")
        switch = SwitchResidueTypeSetMover("fa_standard")
        switch.apply(pose0)


    ########################################################
    # full-atom refinement
    ########################################################

    if args.fastrelax == True:
        mmap = MoveMap()
        mmap.set_bb(True)
        mmap.set_chi(True)
        mmap.set_jump(True)
        
        # First round: Repeat 2 torsion space relax w/ strong disto/anglogram constraints
        sf_fa_round1 = create_score_function('ref2015_cart')
        sf_fa_round1.set_weight(rosetta.core.scoring.atom_pair_constraint, 3.0)
        sf_fa_round1.set_weight(rosetta.core.scoring.dihedral_constraint, 1.0)
        sf_fa_round1.set_weight(rosetta.core.scoring.angle_constraint, 1.0)
        sf_fa_round1.set_weight(rosetta.core.scoring.pro_close, 0.0)
        
        relax_round1 = rosetta.protocols.relax.FastRelax(sf_fa_round1, "%s/data/relax_round1.txt"%scriptdir)
        relax_round1.set_movemap(mmap)
        
        print('relax: First round... (focused on torsion space relaxation)')
        params['PCUT'] = 0.15
        pose0.remove_constraints()
        add_rst(pose0, rst, 3, len(seq), params, nogly=True, use_orient=True)
        relax_round1.apply(pose0)
       
        # Set options for disulfide tolerance -> 0.5A
        print (rosetta.basic.options.get_real_option('in:detect_disulf_tolerance'))
        rosetta.basic.options.set_real_option('in:detect_disulf_tolerance', 0.5)
        print (rosetta.basic.options.get_real_option('in:detect_disulf_tolerance'))

        sf_fa = create_score_function('ref2015_cart')
        sf_fa.set_weight(rosetta.core.scoring.atom_pair_constraint, 0.1)
        sf_fa.set_weight(rosetta.core.scoring.dihedral_constraint, 0.0)
        sf_fa.set_weight(rosetta.core.scoring.angle_constraint, 0.0)
        
        relax_round2 = rosetta.protocols.relax.FastRelax(sf_fa, "%s/data/relax_round2.txt"%scriptdir)
        relax_round2.set_movemap(mmap)
        relax_round2.cartesian(True)
        relax_round2.dualspace(True)

        print('relax: Second round... (cartesian space)')
        params['PCUT'] = 0.30 # To reduce the number of pair restraints..
        pose0.remove_constraints()
        pose0.conformation().detect_disulfides() # detect disulfide bond again w/ stricter cutoffs
        # To reduce the number of constraints, only pair distances are considered w/ higher prob cutoffs
        add_rst(pose0, rst, 3, len(seq), params, nogly=True, use_orient=False, p12_cut=params['PCUT'])
        # Instead, apply CA coordinate constraints to prevent drifting away too much (focus on local refinement?)
        add_crd_rst(pose0, L, std=1.0, tol=2.0)
        relax_round2.apply(pose0)

        # Re-evaluate score w/o any constraints
        scorefxn_min=create_score_function('ref2015_cart')
        scorefxn_min.score(pose0)

    ########################################################
    # save final model
    ########################################################
    pose0.dump_pdb(args.OUT)

if __name__ == '__main__':
    main()
