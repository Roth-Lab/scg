from pipelines.io import make_parent_directory
from pipelines.job_manager import get_job_managers 
from pipelines.ruffus_helpers import run_pipeline
 
from ruffus import *

import gzip
import os
import shutil
import yaml

cwd = os.path.dirname(os.path.realpath(__file__))

#=======================================================================================================================
# Read Command Line Input
#=======================================================================================================================
parser = cmdline.get_argparse(description='''Single cell genotyper.''')

parser.add_argument('--in_file', required=True)

parser.add_argument('--tmp_dir', required=True)

parser.add_argument('--out_dir', required=True,
                    help='''Path where file with posterior probabilities of cluster membership will be written.''')

# Optional arguments
parser.add_argument('--concentration_param', default=1, type=float)

parser.add_argument('--num_restarts', default=1, type=int,
                    help='''Number of random restarts to do. This helps find global optima.''')

# Pipeline arguments
parser.add_argument('--log_dir', default=None,
                    help='''Directory where log files will be written.''')

parser.add_argument('--num_cpus', default=1, type=int,
                    help='''Number of jobs to run.''')

parser.add_argument('--run_local', default=False, action='store_true',
                    help='''If set all commands will be run locally not sent to the cluster.''')

args = parser.parse_args()

#=======================================================================================================================
# Scripts
#=======================================================================================================================
bin_dir = os.path.join(cwd, 'bin')

analysis_script = os.path.join(bin_dir, 'analysis.py')

#=======================================================================================================================
# Pipeline
#=======================================================================================================================
@originate([os.path.join(args.tmp_dir, 'seed', '{0}.txt'.format(i)) for i in range(args.num_restarts)])
def write_seed_dummy_files(out_file):
    make_parent_directory(out_file)
    
    open(out_file, 'w').close()
    
@subdivide(write_seed_dummy_files,
           formatter(),
           inputs((args.in_file)),
           ['{subpath[0][1]}/runs/{basename[0]}/Y.tsv.gz',
            '{subpath[0][1]}/runs/{basename[0]}/Z_0.tsv.gz',
            '{subpath[0][1]}/runs/{basename[0]}/Z_1.tsv.gz',
            '{subpath[0][1]}/runs/{basename[0]}/G.tsv.gz',
            '{subpath[0][1]}/runs/{basename[0]}/params.yaml'],
           '{subpath[0][1]}/runs/{basename[0]}/',
           '{basename[0]}')
def run_analysis(in_file, out_files, out_dir, seed):
    make_parent_directory(out_files[0])
    
    cmd = 'python'
    
    cmd_args = [
                analysis_script,
                '--in_file', in_file,
                '--out_dir', out_dir,
                '--seed', seed,
                '--concentration', args.concentration_param,
                '--convergence_tolerance', 0.001
                ]
    
    run_cmd(cmd, cmd_args)
    
@collate(run_analysis, formatter('params.yaml'), '{subpath[0][1]}/best_run.txt')
def find_best_run(in_files, out_file):
    best_lower_bound = float('-inf')
    
    best_run_id = None
    
    for file_name in in_files:
        with open(file_name) as fh:
            params = yaml.load(fh)
        
        if not params['converged']:
            print file_name, 'failed'
            continue
        
        if params['lower_bound'] > best_lower_bound:
            best_lower_bound = params['lower_bound']
            
            best_run_id = os.path.basename(os.path.dirname(file_name))
    
    with open(out_file, 'w') as fh:
        fh.write(best_run_id)

@transform(find_best_run, formatter(), args.out_dir)
def copy_best_run(in_file, out_dir):
    with open(in_file) as fh:
        best_run_id = fh.readline().strip()
    
    best_run_dir = os.path.join(args.tmp_dir, 'runs', best_run_id)
    
    shutil.copytree(best_run_dir, out_dir)

#=======================================================================================================================
# Run Pipeline
#=======================================================================================================================
if args.log_dir is None:
    log_dir = os.path.join(args.tmp_dir, 'log')

else:
    log_dir = args.log_dir

run_local_cmd, run_cmd, local_job_manager, job_manager = get_job_managers(args.run_local, log_dir)

run_pipeline(args, job_manager, multithread=args.num_cpus)
