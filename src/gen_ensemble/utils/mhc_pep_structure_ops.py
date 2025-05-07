
import Bio
from Bio.PDB import *
from Bio.PDB import PDBParser

def read_structure_and_return_pep(pdbfile):

    p = PDBParser(QUIET=True)
    s = p.get_structure('X', pdbfile)
    pep = ''
    for model in s:
        s.atom_to_internal_coordinates()
        for chain in model:
            if len(chain) < 20:
                pep = chain
                break
    return pep

def read_structure_and_return_pep_from_ensemble(pdbfile):

    p = PDBParser(QUIET=True)
    s = p.get_structure('X', pdbfile)
    pep = []
    for model in s:
        s.atom_to_internal_coordinates()
        for chain in model:
            if len(chain) < 20:
                pep.append(chain)
    return pep

def identify_rigid_pep_residues(pdbfile):

    p = PDBParser(QUIET=True)
    s = p.get_structure('X', pdbfile)

    interface_residues = defaultdict(list)

    pep_chain = None
    mhc_chain = None
    for model in s:
        for chain in model:
            if len(chain) < 20:
                pep_chain = chain
            else:
                mhc_chain = chain

    mhc_atoms  = Bio.PDB.Selection.unfold_entities(mhc_chain, 'A')
    neighbors = Bio.PDB.NeighborSearch(mhc_atoms)

    for res_pep in pep_chain:
        for atom_pep in res_pep:
            
            close_atoms = neighbors.search(atom_pep.coord, 3.0)

            if len(close_atoms) > 0:
                for a in close_atoms:

                    key = res_pep
                    value = a.get_parent()
                    if key in interface_residues:
                        if value not in interface_residues[key]:
                            interface_residues[key].append(value)
                    else:
                        interface_residues[key].append(value)

        if res_pep not in interface_residues:
            interface_residues[res_pep] = []

    #print_string = os.path.basename(pdbfile).replace("_reordered.pdb", "")+","+str(len(interface_residues))
    ignore = []
    i = 0
    for key in interface_residues:
        #print_string += ","+str(len(interface_residues[key]))
        if len(interface_residues[key]) == 0:
            ignore.append(i)

        i += 1

    return ignore

def split_pep_mhc_chains(pdbfile):

    pymol.cmd.load(pdbfile)
    pymol.cmd.do("select resi 181-")
    pymol.cmd.extract("pep", "sele")
    pymol.cmd.alter('pep and chain A', 'chain="B"')
    pymol.cmd.create('merged', "all")
    pymol.cmd.do("save "+pdbfile+", merged")
    pymol.cmd.do("delete all")

def renumber_residues(pdbfile, begin=1):
    """ Renumbers residues in a structure starting from begin (default: 1).
        Keeps numbering consistent with gaps if they exist in the original numbering.
    """

    p = PDBParser()
    structure = p.get_structure('X', pdbfile)
    for model in structure:
        for chain in model:
            if chain.get_list()[0].get_id()[1] == 0:
                return structure
            fresidue_num = chain.get_list()[0].get_id()[1]
            displace = begin - fresidue_num
            for res in chain:
                res.id = (res.id[0], res.id[1]+displace, res.id[2])

    io = PDBIO()
    io.set_structure(structure)
    io.save(pdbfile)