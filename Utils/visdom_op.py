
import visdom
import torch
import numpy as np

def setup_visdom(**kwargs):

    """
    eg :
        vis_eval = setup_visdom(env='SSD_eval')

    :param kwargs:
    :return:
    """
    vis = visdom.Visdom(**kwargs)
    return vis


def visdom_line(vis, y, x, win_name, update='append'):

    """
    eg :
        visdom_line(vis_train, y=[loss], x=iteration, win_name='loss')

    :param vis: created by the setup_visdom function
    :param y: the y-axis data, a series of data that can be passed in at the same time.　eg : [loss1, loss2]
    :param x: X-axis, same format as Y
    :param win_name: the name of the drawing window, must be specified, otherwise it will keep creating windows
    :param update: the drawing method.　The default is append, which is used to record the loss change curve.
    :return.
    """
    if not isinstance(y,torch.Tensor):
        y=torch.Tensor(y)
    y = y.unsqueeze(0)
    x = torch.Tensor(y.size()).fill_(x)
    vis.line(Y=y, X=x, win=win_name, update=update, opts={'title':win_name})
    return True


def visdom_images(vis, images,win_name,num_show=None,nrow=None):
    """
    eg:
        visdom_images(vis_train, images, num_show=3, nrow=3, win_name='Image')

    visdom displays images, by default only 6 images are displayed, 3 per row.

    :param vis: created by the setup_visdom function
    :param images: number of images, shape:[B,N,W,H]
    :param win_name: the name of the drawing window, must be specified, otherwise it will keep creating windows
    :param num_show: the number of images to display, default six
    :param nrow: the number of images to show per line, default three
    :return.
    """
    if not num_show:
        num_show = 6
    if not nrow:
        nrow = 3
    num = images.size(0)
    if num > num_show:
        images = images [:num_show]
    vis.images(tensor=images,nrow=nrow,win=win_name)
    return True


def visdom_image(vis, image,win_name):
    """
    eg :
        visdom_image(vis=vis, image=drawn_image, win_name='image')

    :param vis: created by the setup_visdom function
    :param image: single image tensor, shape:[n,w,h]
    :param win_name: the name of the drawing window, must be specified, otherwise it will keep creating windows
    :return.
    """
    vis.image(img=image, win=win_name)
    return True

def visdom_bar(vis, X, Y, win_name):
    """
    Plotting a bar chart
    eg:
        visdom_bar(vis_train, X=cfg.DATASETS.CLASS_NAME, Y=ap, win_name='ap', title='ap')

    :param vis.
    :param X: category
    :param Y: number
    :param win_name: the name of the drawing window, must be specified, otherwise it will keep creating windows
    :return.
    """
    dic = dict(zip(X,Y))
    del_list = []
    for val in dic:
        if np.isnan(dic[val]):
            del_list.append(val)

    for val in del_list:
        del dic[val]

    vis.bar(X=list(dic.values()),Y=list(dic.keys()),win=win_name, opts={'title':win_name})
    return True