from common import *

from dataset import *
from model import *
from remote import *



def make_df_submit(predict, id):
    id_seqpos = []
    for i in id[0]:
        for t in range(107):
            id_seqpos.append(i + '_%d' % t)
    for i in id[1]:
        for t in range(130):
            id_seqpos.append(i + '_%d' % t)

    p0 = predict[0].transpose(2, 0, 1).reshape(5, -1)
    p1 = predict[1].transpose(2, 0, 1).reshape(5, -1)
    predict = np.hstack([p0, p1]).T

    #---
    print(len(id_seqpos))
    print(predict.shape)

    df_submit = pd.DataFrame(data=predict, index=id_seqpos, columns=target_col)
    df_submit.index.name = 'id_seqpos'
    return df_submit



def do_predict(net, valid_loader, is_mixed_precision):

    valid_predict = []
    valid_target = []
    valid_index = []

    start_timer = timer()
    with torch.no_grad():
        net.eval()

        valid_num = 0
        for t, (seq, target, error, signal_to_noise, index) in enumerate(valid_loader):
            target = target.cuda()
            seq = seq.cuda()

            if is_mixed_precision:
                with amp.autocast():
                    pass
            else:
                predict  = net(seq)


            valid_predict.append(predict.data.cpu().numpy())
            valid_target.append(target.data.cpu().numpy())
            valid_index.extend(index)
            valid_num += len(index)

            #---
            print('\r %8d / %d  %s'%(valid_num, len(valid_loader.sampler),time_to_str(timer() - start_timer,'sec')),end='',flush=True)
            #if valid_num==200*4: break

        assert(valid_num == len(valid_loader.sampler))
        print('')

    #------
    predict = np.concatenate(valid_predict)
    target = np.concatenate(valid_target)
    index = valid_index
    id = valid_loader.dataset.df['id'].values[index]

    return predict, target, id

'''
 

'''

def run_submit():
    mode = 'local' #'remote'
    #mode = 'remote'

    for fold in [0,1,2,3,4]: #[0]: #
        for checkpoint in [8000]:
            name = '%s-%08d_model' %(mode,checkpoint)


            out_dir = '/root/share1/kaggle/2020/open_vaccine/result/simple-bbp-cnn-tx-snr-mcrmse/sn1-fold-%d' % fold
            initial_checkpoint = \
                out_dir + '/checkpoint/%08d_model.pth'%checkpoint #

            is_mixed_precision = False #False #True #

            ## setup  ----------------------------------------

            log = Logger()
            log.open(out_dir+'/log.submit.txt',mode='a')
            os.makedirs(out_dir+'/submit/%s'%mode, exist_ok=True)

            ## net ----------------------------------------
            log.write('** net setting **\n')
            net = Net()
            net = net.cuda()

            f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
            state_dict = f['state_dict']
            net.load_state_dict(state_dict, strict=True)  # True


            ## dataset ----------------------------------------
            if mode=='local':
                train_index, valid_index = make_train_split()

                df = pd.read_json(data_dir+'/train.json', lines=True)
                df = df.loc[valid_index[fold]].reset_index(drop=True)
                dataframe = [df]

            if mode=='remote':
                df = pd.read_json(data_dir+'/test.json', lines=True)

                df0 = df.loc[df.seq_length==107].reset_index(drop=True)
                df1 = df.loc[df.seq_length==130].reset_index(drop=True)
                dataframe = [df0,df1]



            ## start testing here! ##############################################
            predict, target, id = [],[],[]
            for t, df in enumerate(dataframe):

                valid_dataset = RNADataset(df)
                valid_loader = DataLoader(
                    valid_dataset,
                    sampler=SequentialSampler(valid_dataset),
                    batch_size=128,
                    drop_last=False,
                    num_workers=0,
                    pin_memory=True,
                    collate_fn=null_collate
                )
                log.write('t = %d\n'%(t))
                log.write('fold = %d\n'%(fold))
                log.write('valid_dataset : \n%s\n'%(valid_dataset))
                log.write('\n')

                p, t, i = do_predict(net, valid_loader, is_mixed_precision)
                predict.append(p)
                target.append(t)
                id.append(i)


            write_pickle_to_file(out_dir+'/submit/%s/predict.pickle'%name,predict)
            write_pickle_to_file(out_dir+'/submit/%s/target.pickle'%name,target)
            write_pickle_to_file(out_dir+'/submit/%s/id.pickle'%name,id)


            if mode == 'local':
                target = target[0]
                predict = predict[0][:,:68]
                print('predict', predict.shape)
                print('target ', target.shape)
                print('')

                num = len(target)
                index, _ = filter_by_signal_to_noise(np.arange(num).tolist(), valid_loader.dataset.df, 1)
                target = target[index]
                predict = predict[index]

                error = np_mc_rmse(predict, target)



                log.write('fold = %d\n'%(fold))
                log.write('initial_checkpoint = %s\n'%(initial_checkpoint))
                log.write('\tmc_rmse(5) = %0.5f\n'%(error[:5].mean()))
                log.write('\tmc_rmse(3) = %0.5f\n'%(error[:3].mean()))
                for i in range(5):
                    log.write('\t\t%16s = %0.5f\n'%(target_col[i], error[i]))
                log.write('\n')


            if mode == 'remote':
                df_submit = make_df_submit(predict, id)
                print(df_submit)
                print(df_submit.shape)
                df_submit.to_csv(out_dir+'/submit/%s/submission.csv'%name)
                pass


def run_ensemble():

    predict = None
    target = None
    id = None
    num_ensemble = 0

    for submit_dir in [
        '/root/share1/kaggle/2020/open_vaccine/result/simple-bbp-cnn-tx-snr-mcrmse/sn1-fold-0/submit/remote-00008000_model',
        '/root/share1/kaggle/2020/open_vaccine/result/simple-bbp-cnn-tx-snr-mcrmse/sn1-fold-1/submit/remote-00008000_model',
        '/root/share1/kaggle/2020/open_vaccine/result/simple-bbp-cnn-tx-snr-mcrmse/sn1-fold-2/submit/remote-00008000_model',
        '/root/share1/kaggle/2020/open_vaccine/result/simple-bbp-cnn-tx-snr-mcrmse/sn1-fold-3/submit/remote-00008000_model',
        '/root/share1/kaggle/2020/open_vaccine/result/simple-bbp-cnn-tx-snr-mcrmse/sn1-fold-4/submit/remote-00008000_model',

    ]:
        if predict is None:
            predict = read_pickle_from_file(submit_dir+'/predict.pickle')
            target  = read_pickle_from_file(submit_dir+'/target.pickle')
            id = read_pickle_from_file(submit_dir+'/id.pickle')
            num_ensemble = 1
        else:
            for i1,i2, in zip(id, read_pickle_from_file(submit_dir + '/id.pickle')):
                assert(i1 == i2).all()

            p = read_pickle_from_file(submit_dir + '/predict.pickle')
            predict[0] += p[0]
            predict[1] += p[1]
            num_ensemble += 1

    #<todo> clamp to -0.5
    predict[0]  = predict[0] /num_ensemble
    predict[1]  = predict[1] /num_ensemble
    df_submit = make_df_submit(predict, id)
    print(df_submit)
    print(df_submit.shape)
    df_submit.to_csv('/root/share1/kaggle/2020/open_vaccine/result/simple-bbp-cnn-tx-snr-mcrmse/chk80-fold-01234-submission.csv')


#####################################################################
if __name__ == '__main__':
    #run_submit()
    run_ensemble()
