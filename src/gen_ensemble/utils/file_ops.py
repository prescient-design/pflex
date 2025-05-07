

from Bio import SeqIO

from gen_ensemble.constants.constants import *

class FASTA:

    def __init__(self, filename):
        self.filename = filename
        self.mhc_chains = {}
        self.pep_chains = {}
        self.pep_lengths = {}
        self.pep_seq = {}
        self.alleles = {}
        self.mhc_seq = {}

        self.read_fasta()

    def read_fasta(self):
        
        for record in SeqIO.parse(self.filename, "fasta"):
            fields = record.id.split('|')
            pdbid = fields[0]
            chain = fields[1]
            allele = fields[2]
            if len(record.seq) < PEP_LENGTH_UL:
                self.pep_chains[pdbid] = chain
                self.pep_lengths[pdbid] = len(record.seq)
                self.pep_seq[pdbid] = str(record.seq)
            else:
                self.mhc_chains[pdbid] = chain
                self.mhc_seq[pdbid] = str(record.seq)

            self.alleles[pdbid] = allele[0]+"*"+allele[1:len(allele)-2]+":"+allele[len(allele)-2:]

    