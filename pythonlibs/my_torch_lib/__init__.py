import csv
import os

import japanize_matplotlib  # noqa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from pythonlibs.my_torch_lib.coins_cm import make_cm
from pythonlibs.my_torch_lib.coins_ls import make_ls


def modify_network_head(net, num_classes, config_net_name):
    """
    ネットワークの種類に応じて最終層を適切に変更する関数

    Args:
        net: モデルのインスタンス
        num_classes: 出力クラス数
        config_net_name: モデルの名前（設定ファイルから取得される文字列）

    Returns:
        modified_net: 最終層が変更されたモデル
    """
    config_net_name = config_net_name.lower()  # 小文字に変換して比較

    if "vit" in config_net_name:
        try:
            fc_in_features = net.heads.head.in_features
            net.heads.head = nn.Linear(fc_in_features, num_classes)
        except AttributeError as e:
            print(f"ViTの構造が想定と異なります: {e}")
            try:
                # 一部のViTモデルでは異なる構造を持つ可能性がある
                fc_in_features = net.head.in_features
                net.head = nn.Linear(fc_in_features, num_classes)
            except AttributeError:
                raise Exception("サポートされていないViT構造です")

    elif "vgg" in config_net_name:
        try:
            in_features = net.classifier[6].in_features
            net.classifier[6] = nn.Linear(in_features, num_classes)
            net.avgpool = nn.Identity()
        except (AttributeError, IndexError) as e:
            print(f"VGGの構造が想定と異なります: {e}")
            # VGGの別の構造に対する処理を追加することも可能
            raise Exception("サポートされていないVGG構造です")

    elif "resnet" in config_net_name:
        try:
            fc_in_features = net.fc.in_features
            net.fc = nn.Linear(fc_in_features, num_classes)
        except AttributeError as e:
            print(f"ResNetの構造が想定と異なります: {e}")
            raise Exception("サポートされていないResNet構造です")

    else:
        raise ValueError(
            f"サポートされていないネットワーク種類です: {config_net_name}"
        )


# 損失関数値計算用
def eval_loss(loader, device, net, criterion):
    # DataLoaderから最初の1セットを取得する
    for images, labels in loader:
        break

    # デバイスの割り当て
    inputs = images.to(device)
    labels = labels.to(device)

    # 予測値の計算
    outputs = net(inputs)

    #  損失値の計算
    loss = criterion(outputs, labels)

    return loss


# 学習用関数
def fit(
    net,
    optimizer,
    criterion,
    num_epochs,
    classes,
    train_loader,
    test_loader,
    device,
    history,
    program_name,
    save_dir,
    which_data,
    save_model=True,
    save_cm_ls=True,
):
    base_epochs = len(history)

    tensor_classes = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    for epoch in range(base_epochs, num_epochs + base_epochs):
        # 1エポックあたりの正解数(精度計算用)
        n_train_acc = 0
        n_val_acc = np.array([0, 0, 0, 0, 0])
        # 1エポックあたりの累積損失(平均化前)
        train_loss, val_loss = 0, 0
        # 1エポックあたりのデータ累積件数
        n_train, n_test = 0, 0

        # balanced_accuracyを計算するための辞書
        balanced_acc_dict = {
            "label": {str(i1): 0 for i1 in range(0, 15)},
            "train": {str(i1): 0 for i1 in range(0, 15)},
            "label_test": {str(i1): 0 for i1 in range(0, 15)},
            "test": {str(i1): 0 for i1 in range(0, 15)},
            "train_BA": 0,
            "test_BA": 0,
        }

        # 訓練フェーズ
        net.train()

        # train_progress_bar = tqdm(total=len(train_loader), leave=False)
        # for inputs, labels in train_loader:
        for inputs, labels in tqdm(train_loader, leave=False, disable=True):
            # 1バッチあたりのデータ件数
            train_batch_size = len(labels)
            # 1エポックあたりのデータ累積件数
            n_train += train_batch_size

            # GPUヘ転送
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 勾配の初期化
            optimizer.zero_grad()

            # 予測計算
            outputs = net(inputs)

            # 損失計算
            loss = criterion(outputs, labels)

            # 勾配計算
            loss.backward()

            # パラメータ修正
            optimizer.step()

            # 予測ラベル導出
            predicted = torch.max(outputs, 1)[1]

            # 平均前の損失と正解数の計算
            # lossは平均計算が行われているので平均前の損失に戻して加算
            train_loss += loss.item() * train_batch_size
            n_train_acc += (predicted == labels).sum().item()

            # trainデータの予測ラベルに対して、各ラベルごとの出現回数をカウント
            for i in range(len(labels)):
                balanced_acc_dict["label"][str(labels[i].item())] += 1
                if predicted[i].item() == labels[i].item():
                    balanced_acc_dict["train"][str(predicted[i].item())] += 1

        # 予測フェーズ
        net.eval()

        for inputs_test, labels_test in test_loader:
            # 1バッチあたりのデータ件数
            test_batch_size = len(labels_test)
            # 1エポックあたりのデータ累積件数
            n_test += test_batch_size

            # GPUヘ転送
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)

            # 予測計算
            outputs_test = net(inputs_test)

            # 損失計算
            loss_test = criterion(outputs_test, labels_test)

            # 予測ラベル導出
            predicted_test = torch.max(outputs_test, 1)[1]

            #  平均前の損失と正解数(4bit中何個正しいか)の計算
            # lossは平均計算が行われているので平均前の損失に戻して加算
            val_loss += loss_test.item() * test_batch_size
            for i in range(len(labels_test)):
                correct = (
                    (
                        tensor_classes[predicted_test[i]]
                        == tensor_classes[labels_test[i]]
                    )
                    .sum()
                    .item()
                )
                n_val_acc[correct] += 1

            # testデータの予測ラベルについて、各ラベルごとの出現回数をカウント
            for i in range(len(labels_test)):
                balanced_acc_dict["label_test"][
                    str(labels_test[i].item())
                ] += 1
                if predicted_test[i].item() == labels_test[i].item():
                    balanced_acc_dict["test"][
                        str(predicted_test[i].item())
                    ] += 1

        # # Close the progress bars
        # train_progress_bar.close()

        # 精度計算
        train_acc = n_train_acc / n_train
        val_acc = n_val_acc / n_test
        # 損失計算
        avg_train_loss = train_loss / n_train
        avg_val_loss = val_loss / n_test

        # balanced_accuracy計算
        non_zero_label = 0
        for i in range(15):
            # データセットに存在しないラベルの場合は0を加算
            if balanced_acc_dict["label"][str(i)] == 0:
                balanced_acc_dict["train_BA"] += 0
            else:
                non_zero_label += 1
                balanced_acc_dict["train_BA"] += (
                    balanced_acc_dict["train"][str(i)]
                    / balanced_acc_dict["label"][str(i)]
                )

            # データセットに存在しないラベルの場合は0を加算
            if balanced_acc_dict["label_test"][str(i)] == 0:
                balanced_acc_dict["test_BA"] += 0
            else:
                balanced_acc_dict["test_BA"] += (
                    balanced_acc_dict["test"][str(i)]
                    / balanced_acc_dict["label_test"][str(i)]
                )
        balanced_acc_dict["train_BA"] /= non_zero_label
        balanced_acc_dict["test_BA"] /= non_zero_label

        # 結果表示
        print(
            f"Epoch [{(epoch+1)}/{num_epochs+base_epochs}]:"
            f"loss: {avg_train_loss:.5f}, acc: {train_acc:.5f}, BA: {balanced_acc_dict['train_BA']:.5f}, "
            f"val_loss: {avg_val_loss:.5f}, val_acc: {val_acc[4]:.5f}, val_BA: {balanced_acc_dict['test_BA']:.5f}"
        )
        # 記録
        item = np.array(
            [
                epoch + 1,
                avg_train_loss,
                train_acc,
                avg_val_loss,
                *val_acc,
                balanced_acc_dict["train_BA"],
                balanced_acc_dict["test_BA"],
            ]
        )
        history = np.vstack((history, item))

        # モデルを保存
        if (epoch + 1) == num_epochs:
            if save_model is True:
                torch.save(
                    net,
                    os.path.join(
                        save_dir,
                        f"epoch{epoch}.pth",
                    ),
                )

        if epoch % 25 == 0 or (epoch + 1) == num_epochs:
            if save_cm_ls is True:
                make_cm(device, epoch, classes, test_loader, save_dir, net)
                make_ls(device, epoch, test_loader, save_dir, net)

    return history


def comp_val_acc(
    net1,
    net2,
    optimizer1,
    optimizer2,
    criterion,
    num_epochs,
    train_loader1,
    train_loader2,
    test_loader1,
    device,
    history,
    program_name,
    save_dir,
    which_data,
):
    base_epochs = len(history)

    classes = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    for epoch in range(base_epochs, num_epochs + base_epochs):
        # 1エポックあたりの正解数(精度計算用)
        n_train_acc1 = 0
        n_val_acc1 = np.array([0, 0, 0, 0, 0])
        # 1エポックあたりの累積損失(平均化前)
        train_loss1, val_loss1 = 0, 0
        # 1エポックあたりのデータ累積件数
        n_train1, n_test1 = 0, 0

        n_train_acc2 = 0
        n_val_acc2 = np.array([0, 0, 0, 0, 0])
        # 1エポックあたりの累積損失(平均化前)
        train_loss2, val_loss2 = 0, 0
        # 1エポックあたりのデータ累積件数
        n_train2, n_test2 = 0, 0

        # 訓練フェーズ1
        net1.train()

        # train_progress_bar = tqdm(total=len(train_loader), leave=False)
        # for inputs, labels in train_loader:
        for inputs, labels in tqdm(train_loader1, leave=False, disable=False):
            # 1バッチあたりのデータ件数
            train_batch_size = len(labels)
            # 1エポックあたりのデータ累積件数
            n_train1 += train_batch_size

            # GPUヘ転送
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 勾配の初期化
            optimizer1.zero_grad()

            # 予測計算
            outputs = net1(inputs)

            # 損失計算
            loss = criterion(outputs, labels)

            # 勾配計算
            loss.backward()

            # パラメータ修正
            optimizer1.step()

            # 予測ラベル導出
            predicted = torch.max(outputs, 1)[1]

            # 平均前の損失と正解数の計算
            # lossは平均計算が行われているので平均前の損失に戻して加算
            train_loss1 += loss.item() * train_batch_size
            n_train_acc1 += (predicted == labels).sum().item()

            # # Update the progress bar manually
            # train_progress_bar.update(1)

        # 訓練フェーズ2
        net2.train()

        # train_progress_bar = tqdm(total=len(train_loader), leave=False)
        # for inputs, labels in train_loader:
        for inputs, labels in tqdm(train_loader2, leave=False, disable=False):
            # 1バッチあたりのデータ件数
            train_batch_size = len(labels)
            # 1エポックあたりのデータ累積件数
            n_train2 += train_batch_size

            # GPUヘ転送
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 勾配の初期化
            optimizer2.zero_grad()

            # 予測計算
            outputs = net2(inputs)

            # 損失計算
            loss = criterion(outputs, labels)

            # 勾配計算
            loss.backward()

            # パラメータ修正
            optimizer2.step()

            # 予測ラベル導出
            predicted = torch.max(outputs, 1)[1]

            # 平均前の損失と正解数の計算
            # lossは平均計算が行われているので平均前の損失に戻して加算
            train_loss2 += loss.item() * train_batch_size
            n_train_acc2 += (predicted == labels).sum().item()

            # # Update the progress bar manually
            # train_progress_bar.update(1)

        # 予測フェーズ1
        net1.eval()

        for inputs_test, labels_test in test_loader1:
            # 1バッチあたりのデータ件数
            test_batch_size = len(labels_test)
            # 1エポックあたりのデータ累積件数
            n_test1 += test_batch_size

            # GPUヘ転送
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)

            # 予測計算
            outputs_test = net1(inputs_test)

            # 損失計算
            loss_test = criterion(outputs_test, labels_test)

            # 予測ラベル導出
            predicted_test = torch.max(outputs_test, 1)[1]

            #  平均前の損失と正解数の計算
            # lossは平均計算が行われているので平均前の損失に戻して加算
            val_loss1 += loss_test.item() * test_batch_size
            for i in range(len(labels_test)):
                correct = (
                    (classes[predicted_test[i]] == classes[labels_test[i]])
                    .sum()
                    .item()
                )
                n_val_acc1[correct] += 1

        # # Close the progress bars
        # train_progress_bar.close()

        # 予測フェーズ2
        net2.eval()

        for inputs_test, labels_test in test_loader1:
            # 1バッチあたりのデータ件数
            test_batch_size = len(labels_test)
            # 1エポックあたりのデータ累積件数
            n_test2 += test_batch_size

            # GPUヘ転送
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)

            # 予測計算
            outputs_test = net2(inputs_test)

            # 損失計算
            loss_test = criterion(outputs_test, labels_test)

            # 予測ラベル導出
            predicted_test = torch.max(outputs_test, 1)[1]

            #  平均前の損失と正解数の計算
            # lossは平均計算が行われているので平均前の損失に戻して加算
            val_loss2 += loss_test.item() * test_batch_size
            for i in range(len(labels_test)):
                correct = (
                    (classes[predicted_test[i]] == classes[labels_test[i]])
                    .sum()
                    .item()
                )
                n_val_acc2[correct] += 1

        # # Close the progress bars
        # train_progress_bar.close()

        # 精度計算
        train_acc1 = n_train_acc1 / n_train1
        val_acc1 = n_val_acc1 / n_test1
        # 損失計算
        avg_train_loss1 = train_loss1 / n_train1
        avg_val_loss1 = val_loss1 / n_test1

        # 精度計算
        train_acc2 = n_train_acc2 / n_train2
        val_acc2 = n_val_acc2 / n_test2
        # 損失計算
        avg_train_loss2 = train_loss2 / n_train2
        avg_val_loss2 = val_loss2 / n_test2

        # 結果表示
        print(
            f"Epoch [{(epoch+1)}/{num_epochs+base_epochs}],\n"
            f"loss1: {avg_train_loss1:.5f} acc1: {train_acc1:.5f} "
            f"val_loss1: {avg_val_loss1:.5f}, val_acc1: {val_acc1[4]:.5f},\n"
            f"loss2: {avg_train_loss2:.5f} acc2: {train_acc2:.5f} ",
            f"val_loss2: {avg_val_loss2:.5f}, val_acc2: {val_acc2[4]:.5f}",
        )
        # 記録
        item = np.array(
            [
                epoch + 1,
                avg_train_loss1,
                train_acc1,
                avg_val_loss1,
                *val_acc1,
                avg_train_loss2,
                train_acc2,
                avg_val_loss2,
                *val_acc2,
            ]
        )
        history = np.vstack((history, item))

        # if epoch % 25 == 0 or epoch == num_epochs:
        #     make_cm(device, epoch, test_loader, save_dir, net)
        #     make_ls(device, epoch, test_loader, save_dir, net)

    return history


# 学習用関数
def fit_vec(
    net,
    optimizer,
    criterion,
    num_epochs,
    train_loader,
    test_loader,
    device,
    history,
    program_name,
    save_dir,
    which_data,
    save_model=True,
    save_cm_ls=True,
):
    base_epochs = len(history)

    classes = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    for epoch in range(base_epochs, num_epochs + base_epochs):
        # 1エポックあたりの正解数(精度計算用)
        n_train_acc = 0
        n_val_acc = np.array([0, 0, 0, 0, 0])
        # 1エポックあたりの累積損失(平均化前)
        train_loss, val_loss = 0, 0
        # 1エポックあたりのデータ累積件数
        n_train, n_test = 0, 0

        # 訓練フェーズ
        net.train()

        # train_progress_bar = tqdm(total=len(train_loader), leave=False)
        # for inputs, labels in train_loader:
        for inputs, labels in tqdm(train_loader, leave=False, disable=False):
            # 1バッチあたりのデータ件数
            train_batch_size = len(labels)
            # 1エポックあたりのデータ累積件数
            n_train += train_batch_size

            # GPUヘ転送
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 勾配の初期化
            optimizer.zero_grad()

            # 予測計算
            outputs = net(inputs)

            labels_vec = torch.tensor(
                [classes[labels[i].item()] for i in range(len(labels))]
            ).to(device)
            # print(labels_vec)
            outputs_sig = torch.sigmoid(outputs)
            # 損失計算
            loss = criterion(outputs_sig, labels_vec) * len(classes[0])

            # 勾配計算
            loss.backward()

            # パラメータ修正
            optimizer.step()

            # 予測ラベル導出
            predicted = torch.where(outputs_sig < 0.5, 0.0, 1.0)

            # 平均前の損失と正解数の計算
            # lossは平均計算が行われているので平均前の損失に戻して加算
            train_loss += loss.item() * train_batch_size
            # n_train_acc += (predicted == labels).sum().item()
            for i in range(len(labels)):
                correct = (predicted[i] == labels_vec[i]).sum().item()
                if correct == 4:
                    n_train_acc += 1

            # # Update the progress bar manually
            # train_progress_bar.update(1)

        # 予測フェーズ
        net.eval()

        for inputs_test, labels_test in test_loader:
            # 1バッチあたりのデータ件数
            test_batch_size = len(labels_test)
            # 1エポックあたりのデータ累積件数
            n_test += test_batch_size

            # GPUヘ転送
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)

            # 予測計算
            outputs_test = net(inputs_test)

            # 損失計算
            outputs_test_sig = torch.sigmoid(outputs_test)
            labels_test_vec = torch.tensor(
                [
                    classes[labels_test[i].item()]
                    for i in range(len(labels_test))
                ]
            ).to(device)
            loss_test = criterion(outputs_test_sig, labels_test_vec) * len(
                classes[0]
            )

            # 予測ラベル導出
            predicted_test = torch.where(outputs_test_sig < 0.5, 0.0, 1.0)

            #  平均前の損失と正解数の計算
            # lossは平均計算が行われているので平均前の損失に戻して加算
            val_loss += loss_test.item() * test_batch_size
            for i in range(len(labels_test)):
                correct = (
                    (classes[predicted_test[i]] == classes[labels_test[i]])
                    .sum()
                    .item()
                )
                n_val_acc[correct] += 1

        # # Close the progress bars
        # train_progress_bar.close()

        # 精度計算
        train_acc = n_train_acc / n_train
        val_acc = n_val_acc / n_test
        # 損失計算
        avg_train_loss = train_loss / n_train
        avg_val_loss = val_loss / n_test
        # 結果表示
        print(
            f"Epoch [{(epoch+1)}/{num_epochs+base_epochs}],"
            f"loss: {avg_train_loss:.5f} acc: {train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {val_acc:.5f}"
        )
        # 記録
        item = np.array(
            [epoch + 1, avg_train_loss, train_acc, avg_val_loss, *val_acc]
        )
        history = np.vstack((history, item))

        # モデルを保存
        if epoch == num_epochs:
            if save_model is True:
                torch.save(
                    net,
                    os.path.join(
                        os.path.expanduser("~"),
                        "static",
                        f"{which_data}",
                        f"{program_name}",
                        f"epoch{epoch}.pth",
                    ),
                )

        if epoch % 25 == 0 or epoch == num_epochs:
            if save_cm_ls is True:
                make_cm(device, epoch, test_loader, save_dir, net)
                make_ls(device, epoch, test_loader, save_dir, net)

    return history


# 学習ログ解析
def evaluate_history(history, save_dir, data_name=None):
    # 損失と精度の確認
    max_index = history[:, 8].argmax()

    result_f = open(
        f"{save_dir}/abst.txt",
        "a",
        newline="\n",
    )
    datalines = [
        f"使用した訓練データ: {data_name}\n\n",
        "検証データの成績\n",
        f"初期状態: 損失: {history[0,3]:.5f} 精度: {history[0,8]:.5f} BA: {history[0,10]:.5f}\n",
        f"最終状態: 損失: {history[-1,3]:.5f} 精度: {history[-1,8]:.5f} BA: {history[-1,10]:.5f}\n",
        f"max(精度): 損失: {history[max_index,3]:.5f} 精度: {history[max_index,8]:.5f} BA: {history[max_index,10]:.5f}\n",
    ]
    result_f.writelines(datalines)
    result_f.close()

    num_epochs = len(history)
    if num_epochs < 10:
        unit = 1
    else:
        unit = num_epochs / 10

    # 学習曲線の表示 (損失)
    plt.figure(figsize=(9, 8))
    plt.plot(history[:, 0], history[:, 1], "b", label="訓練")
    plt.plot(history[:, 0], history[:, 3], "k", label="検証")
    plt.xticks(np.arange(0, num_epochs + 1, unit))
    plt.xlabel("繰り返し回数")
    plt.ylabel("損失")
    plt.title("学習曲線(損失)")
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{data_name}_loss.png"))
    plt.show()

    # 学習曲線の表示 (精度)
    plt.figure(figsize=(9, 8))
    plt.plot(history[:, 0], history[:, 2], "b", label="訓練")
    plt.plot(history[:, 0], history[:, 8], "k", label="検証")
    plt.plot(history[:, 0], history[:, 7], "g", label="1miss")
    plt.plot(history[:, 0], history[:, 6], "c", label="2miss")
    plt.plot(history[:, 0], history[:, 5], "y", label="3miss")
    plt.plot(history[:, 0], history[:, 4], "m", label="4miss")
    plt.xticks(np.arange(0, num_epochs + 1, unit))
    plt.xlabel("繰り返し回数")
    plt.ylabel("精度")
    plt.title("学習曲線(精度)")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{data_name}_acc.png"))
    plt.show()


def save_history_to_csv(history, save_dir):
    # 配列をCSVに保存する関数
    filename = os.path.join(save_dir, "history.csv")

    with open(filename, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)

        # 配列をCSVファイルに書き込む
        for row in history:
            csvwriter.writerow(row)

    print(f"配列が {filename} に保存されました。")


# 学習ログ解析
def evaluate_vib_history(history, program_name, save_dir):
    # 損失と精度の確認
    max_index = history[:, 8].argmax()
    max_index2 = history[:, 16].argmax()

    result_f = open(
        f"{save_dir}/{program_name}_abst.txt",
        "w",
        newline="\n",
    )
    datalines = [
        f"初期状態_all_cases: 損失: {history[0,3]:.5f} 精度: {history[0,8]:.5f}\n",
        f"最終状態_all_cases: 損失: {history[-1,3]:.5f} 精度: {history[-1,8]:.5f}\n",
        f"max_all_cases: 損失: {history[max_index,3]:.5f} 精度: {history[max_index,8]:.5f}\n",
    ]
    result_f.writelines(datalines)
    datalines = [
        f"初期状態_1case: 損失: {history[0,11]:.5f} 精度: {history[0,16]:.5f}\n",
        f"最終状態_1case: 損失: {history[-1,11]:.5f} 精度: {history[-1,16]:.5f}\n",
        f"max_1case: 損失: {history[max_index2,11]:.5f} 精度: {history[max_index2,16]:.5f}\n",
    ]
    result_f.writelines(datalines)
    result_f.close()

    num_epochs = len(history)
    if num_epochs < 10:
        unit = 1
    else:
        unit = num_epochs / 10

    # 学習曲線の表示 (損失)
    plt.figure(figsize=(9, 8))
    plt.plot(history[:, 0], history[:, 1], "b", label="訓練_all_cases")
    plt.plot(history[:, 0], history[:, 3], "c", label="検証_all_cases")
    plt.plot(history[:, 0], history[:, 9], "r", label="訓練_1case")
    plt.plot(history[:, 0], history[:, 11], "m", label="検証_1case")
    plt.xticks(np.arange(0, num_epochs + 1, unit))
    plt.xlabel("繰り返し回数")
    plt.ylabel("損失")
    plt.title("学習曲線(損失)")
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{program_name}_loss.png"))
    plt.show()

    # 学習曲線の表示 (精度)
    plt.figure(figsize=(9, 8))
    plt.plot(history[:, 0], history[:, 2], "b", label="訓練_all_cases")
    plt.plot(history[:, 0], history[:, 8], "c", label="検証_all_cases")
    plt.plot(history[:, 0], history[:, 10], "r", label="訓練_1case")
    plt.plot(history[:, 0], history[:, 16], "m", label="検証_1case")
    plt.xticks(np.arange(0, num_epochs + 1, unit))
    plt.xlabel("繰り返し回数")
    plt.ylabel("精度")
    plt.title("学習曲線(精度)")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{program_name}_acc.png"))
    plt.show()


# イメージとラベル表示
def show_images_labels(loader, classes, net, device, program_name, save_dir):
    # DataLoaderから最初の1セットを取得する
    for images, labels in loader:
        break
    # 表示数は50個とバッチサイズのうち小さい方
    n_size = min(len(images), 50)

    if net is not None:
        # デバイスの割り当て
        inputs = images.to(device)
        labels = labels.to(device)

        # 予測計算
        outputs = net(inputs)
        predicted = torch.max(outputs, 1)[1]
        # images = images.to('cpu')

    # 最初のn_size個の表示
    plt.figure(figsize=(20, 15))
    for i in range(n_size):
        ax = plt.subplot(5, 10, i + 1)
        label_name = classes[labels[i]]
        # netがNoneでない場合は、予測結果もタイトルに表示する
        if net is not None:
            predicted_name = classes[predicted[i]]
            # 正解かどうかで色分けをする
            if label_name == predicted_name:
                c = "k"
            else:
                c = "b"
            ax.set_title(label_name + ":" + predicted_name, c=c, fontsize=20)
        # netがNoneの場合は、正解ラベルのみ表示
        else:
            ax.set_title(label_name, fontsize=20)
        # TensorをNumPyに変換
        image_np = images[i].numpy().copy()
        # 軸の順番変更 (channel, row, column) -> (row, column, channel)
        img = np.transpose(image_np, (1, 2, 0))
        # 値の範囲を[-1, 1] -> [0, 1]に戻す
        img = (img + 1) / 2
        # 結果表示
        plt.imshow(img)
        ax.set_axis_off()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{program_name}_images.png"))
    plt.show()


# 間違えた画像を表示
def show_incorrect_images_labels(loader, classes, net, device, save_dir):
    incorrect_i = 0
    image_count = 0
    # DataLoaderから最初の1セットを取得する
    incorrect_count = 0
    all_count = 0
    for images, labels in loader:
        n_size = len(images)
        all_count += n_size

        if net is not None:
            # デバイスの割り当て
            inputs = images.to(device)
            labels = labels.to(device)

            # 予測計算
            outputs = net(inputs)
            predicted = torch.max(outputs, 1)[1]
            images = images.to("cpu")

        for i in range(n_size):
            if incorrect_i % 15 == 0:
                if incorrect_i == 15:
                    plt.savefig(
                        os.path.join(
                            save_dir, f"incorrect_images_{image_count}.png"
                        )
                    )
                    plt.clf()
                    plt.close()
                    image_count += 1
                incorrect_i = 0
                plt.figure(figsize=(20, 15))
            label_name = classes[labels[i]]
            # netがNoneでない場合は、予測結果もタイトルに表示する
            predicted_name = classes[predicted[i]]
            # 正解かどうかで色分けをする
            if label_name != predicted_name:
                incorrect_count += 1
                ax = plt.subplot(3, 5, incorrect_i + 1)
                incorrect_i += 1
                c = "b"
                ax.set_title(
                    label_name + ":" + predicted_name, c=c, fontsize=20
                )

                # TensorをNumPyに変換
                image_np = images[i].cpu().numpy().copy()
                # 軸の順番変更 (channel, row, column) -> (row, column, channel)
                img = np.transpose(image_np, (1, 2, 0))
                # 値の範囲を[-1, 1] -> [0, 1]に戻す
                img = (img + 1) / 2
                # 結果表示
                plt.imshow(img)
                ax.set_axis_off()
    plt.savefig(os.path.join(save_dir, f"incorrect_images_{image_count}.png"))
    plt.clf()
    plt.close()
    plt.show()

    result_f = open(
        f"{save_dir}/abst.txt",
        "a",
        newline="\n",
    )

    datalines = [
        "\n" f"総データ数: {all_count}\n",
        f"間違えた画像の数: {incorrect_count}\n",
    ]
    result_f.writelines(datalines)
    result_f.close()


# PyTorch乱数固定用
def torch_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
