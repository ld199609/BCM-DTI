"""
SPP_DTI
"""
import os, datetime,sys,copy
sys.path.append("../..")
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc
import json

from SSCNN_dataset import *
from SSCNN_model import *
from SSCNN_config import *
from SSCNN_utils import *

if torch.cuda.is_available():
    device = torch.device("cuda")  # "cuda:0"
else:
    device = torch.device("cpu")


def test(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (d,f, p, label) in enumerate(data_generator):
        score = model(d.long().to(device),f.long().to(device), p.long().to(device))

        logits = torch.squeeze(score)
        loss_fct = torch.nn.BCELoss()
        np.seterr(divide='ignore',invalid='ignore')
        label = Variable(torch.from_numpy(np.array(label)).float()).to(device)

        loss = loss_fct(logits, label)

        loss_accumulate += loss
        count += 1

        logits = logits.detach().cpu().numpy()

        label_ids = label.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()

    loss = loss_accumulate / count

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)

    precision = tpr / (tpr + fpr)

    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)

    thred_optim = thresholds[5:][np.argmax(f1[5:])]

    # print("optimal threshold: " + str(thred_optim))

    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]

    auc_k = auc(fpr, tpr)
    # print("AUROC:" + str(auc_k))
    # print("AUPRC: " + str(average_precision_score(y_label, y_pred)))

    cm1 = confusion_matrix(y_label, y_pred_s)
    recall= recall_score(y_label, y_pred_s)
    precision=precision_score(y_label, y_pred_s)
    # print('Confusion Matrix : \n', cm1)
    # print('Recall : ', recall_score(y_label, y_pred_s))
    # print('Precision : ', precision_score(y_label, y_pred_s))

    total1 = sum(sum(cm1))
    #####from confusion matrix calculate accuracy
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    # print('Accuracy : ', accuracy1)

    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    # print('Sensitivity : ', sensitivity1)

    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    # print('Specificity : ', specificity1)

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred),accuracy1,recall,precision, f1_score(y_label,
                                                                                              outputs), y_pred, loss.item()

def formula_output(prefix,epoch,auc,auprc,p,r,f1,loss):
    return dataset_name+" "+prefix+" at"+" Epoch"+epoch+" AUC: "+auc+" AUCPR: "+auprc+" Precision: "+p+" Recall: "+r+" F1: "+f1+" Loss: "+loss

def train(config):
    loss_history = []
    best_auc=0
    counter = 0
    model = SSCNN_DTI(config)

    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    model_max = copy.deepcopy(model)

    with torch.set_grad_enabled(False):
        auc, auprc,acc,p,recall, f1, logits, loss = test(config['test_loader'], model_max)
        print(formula_output("Testing","-1",str(auc),str(auprc),str(p),str(recall),str(f1),str(loss)))


    print('--- Go for Training ---')
    torch.backends.cudnn.benchmark = True
    for epo in range(config['epochs']):
        model.train()
        for i, (d, f, p, label) in enumerate(train_loader):
            score = model(d.long().to(device),f.long().to(device),p.long().to(device))
            np.seterr(divide='ignore',invalid='ignore')
            label = Variable(torch.from_numpy(np.array(label)).float()).to(device)

            loss_fct = torch.nn.BCELoss()
            # m = torch.nn.Sigmoid()
            n = torch.squeeze(score)

            loss = loss_fct(n, label)
            loss_history.append(loss)

            opt.zero_grad()
            loss.backward()
            if config['clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

            if (i % 1000 == 0):
                print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(
                    loss.cpu().detach().numpy()))

        # every epoch test
        with torch.set_grad_enabled(False):
            auc, auprc,acc,p,recall,  f1, logits, loss = test(test_loader, model)
            if acc > best_auc and epo >= 5:
                best_auc = acc
                counter = 0
                model_max = copy.deepcopy(model)
                save_dir = './model/' + dataset_name + "/"
                assert_dir_exist(save_dir)
                print("new model saved")
                save_best_model(model, model_dir=save_dir, best_epoch=i)
            else:
                counter += 1
            if counter == args['stop_counter']:
                print("early stop at epoch %d" % epo)
                break

            print(formula_output("Validation", str(epo + 1), str(auc),str(auprc), str(p), str(recall), str(f1),
                                 "not record"))

    print('--- Go for Testing ---')
    try:
        with torch.set_grad_enabled(False):
            auc,auprc,acc,p,recall, f1, logits, loss = test(test_loader, model_max)
            print(formula_output("Testing", "", str(auc), str(auprc),str(p), str(recall), str(f1),
                                 str(loss)))
    except:
        print('testing failed')
    return model_max, loss_history


if __name__ == "__main__":

    best_output=[]
    # last_auc=best_model

    dataset_name = 'BIOSNAP'
    # print(dataset_config)
    dataset = dataset_config[dataset_name][0]
    root_path = dataset_config[dataset_name][0]
    input_dir = dataset_config['dir'][1]
    output_dir = dataset_config["dir"][2]

    decompose='bcm'
    input_path = "../../"+root_path + "/" + input_dir
    output_path = "../"+root_path + "/" + output_dir

    trainSmiles1, trainSmiles2, trainLabel,testSmiles1, testSmiles2, testLabel=load_bcm_dataset(input_path)
    frag_set = list(set(l for s in trainSmiles1+trainSmiles2+testSmiles1+testSmiles2 for l in s))

    frag_len=[len(d) for d in trainSmiles1+trainSmiles2+testSmiles1+testSmiles2]

    words2idx_d = dict(zip(frag_set, range(0, len(frag_set))))

    args=SSCNN_args()


    args['max_drug_seq']= max(frag_len)
    # args['max_protein_seq']= max(frag_len_p)
    args['input_d_dim']=len(frag_set)+1
    # args['input_p_dim']=len(frag_len_p)+1

    args['d_channel_size'][0]=args['max_drug_seq']
    args['p_channel_size'][0]=args['max_protein_seq']

    trainDataset = NewDataset(trainSmiles1, trainSmiles2, trainLabel,words2idx_d,args['max_drug_seq'])
    testDataset = NewDataset(testSmiles1, testSmiles2, testLabel,words2idx_d,args['max_drug_seq'])


    train_loader = DataLoader(dataset=trainDataset, batch_size=args['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=testDataset, batch_size=args['batch_size'], shuffle=True, drop_last=True)

    args['train_loader'] = train_loader
    args['test_loader'] = test_loader

    # print(args)
    model_max,_=train(args)
    flops=params_count(model_max)
    print(flops,decompose,dataset_name)



