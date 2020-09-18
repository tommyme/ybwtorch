from utils.dataloader import tta_trans_list

def calc_acc(data_loader,model,num_of_trans,device):
    pred_list = [list() for i in range(num_of_trans)]
    target_list = []
    for img_set, target in data_loader:
        for i,imgs in enumerate(img_set):
            pred = model(imgs.to(device))
            pred = pred.argmax(dim=1)
            pred_list[i]+=(pred)
        target_list.+=(target)
    result,_ = torch.mode(torch.tensor(pred_list),dim = 0)
    correct = torch.eq(result,torch.tensor(target_list)).sum().cpu().numpy()

    acc = 100. * correct / data_loader.dataset.__len__()
    batch_eval_info = "{}/{}  test_acc: {:.2f}%".format(
         correct, data_loader.dataset.__len__(),acc)
    print(batch_eval_info)
    return acc
	

def get_best_tta_list(tta_trans_list,data_loader,model,device):
    lists = []
    accs = []
    for i in range(0,len(tta_trans_list)+1): # 1-6 子集大小
        for j in range(len(tta_trans_list)+1-i): #start point
            lists.append(tta_trans_list[j:j+i])
    num_of_trans = len(tta_trans_list)
    for each in lists:
        accs.append(calc_acc(data_loader,model,num_of_trans,device))
    
    idx = accs.index(max(accs))
    
    return lists[idx]
        