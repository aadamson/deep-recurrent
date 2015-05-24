import subprocess
import os
import json
import argparse
import random
import jinja2

templateLoader = jinja2.FileSystemLoader( searchpath="templates" )
template_env = jinja2.Environment( loader=templateLoader )

def stream_run_script(options):
    template = template_env.get_template('run.jinja')
    script = template.stream(model=options['model'],
                             data=options['data'],
                             lam=options['lambda'],
                             emb_file=options['emb_file'],
                             num_tokens=options['num_tokens'],
                             emb_dim=options['emb_dim'],
                             weight=options['weight'],
                             lr=options['lr'],
                             dr=options['dr'],
                             outdir=options['outdir'])
    return script

def render_job_script(options):
    template = template_env.get_template('qsub.jinja')
    script = template.render(outdir=options['outdir'],
                             model_name=options['model_name'],
                             script_path=options['script_path'])

    return script


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', dest='outdir', type=str,
                        help='Output directory for run')
    parser.add_argument('-l', '--lambda', dest='lam', type=float, default=0.0001,
                        help='Set lambda value used for L2-regularization')
    parser.add_argument('--emb-file', dest='emb_file', type=str,
                        help='Location of file containing word embeddings')
    parser.add_argument('-nt', dest='num_tokens', type=int,
                        help='Number of tokens in embeddings file')
    parser.add_argument('-nx', dest='emb_dim', type=int,
                        help='Dimension of each word vector in embeddings file')
    parser.add_argument('--data', dest='data', type=str,
                        help='File containing data to be trained and tested on')
    parser.add_argument('--weight', dest='weight', type=float, nargs='+', default=[0.5],
                        help='Null class weights to test on')
    parser.add_argument('-lr', dest='lr', type=float, nargs='+', default=[1e-5],
                        help='Null class weights to test on')
    parser.add_argument('-dr', dest='dr', type=float, nargs='+', default=[0.0],
                        help='Dropout probabilities to test on')
    parser.add_argument('--model', dest='model', type=str, default='GruDeepRecurrent',
                        help='Name of binary of model to run')
    args = parser.parse_args()

    base_dict = { 'emb_file': args.emb_file, 'emb_dim': args.emb_dim, 
                  'num_tokens': args.num_tokens, 'data': args.data,
                  'model': args.model, 'outdir': args.outdir, 'lambda': args.lam }
    options_dicts = []

    for weight in args.weight:
        for lr in args.lr:
            for dr in args.dr:
                new_dict = base_dict.copy()
                new_dict['weight'] = weight
                new_dict['lr'] = lr
                new_dict['dr'] = dr

                options_dicts.append(new_dict)

    for options in options_dicts:
        script = stream_run_script(options)

        model_short_name = options['model'][options['model'].find('/')+1:]
        data_short_name = options['data'][options['data'].rfind('/')+1:options['data'].rfind('.')]
        model_name = "%s_lr_%g_dr_%g_weight_%g_lambda_%g_data_%s" % (model_short_name, options['lr'], options['dr'], options['weight'], options['lambda'], data_short_name)

        script_name = "./temp_scripts/%d.sh" % random.randint(1, 1e6)
        script.dump(script_name)
        subprocess.call(["chmod", "u+x", script_name])

        job_script = render_job_script({ 'outdir': args.outdir, 
                                         'model_name': model_name, 
                                         'script_path': script_name })
        print job_script
        subprocess.call(job_script)


if __name__ == '__main__':
  main()
