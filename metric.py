import logging
import torch
import torch.nn as nn
import sys
import copy
import util
import numpy as np

thismodule = sys.modules[__name__]
logger = logging.getLogger('Classification')

def metric(opt, net, dataloader):
	current_metric = getattr(thismodule, "metric{}".format(opt['current_model']))
	return current_metric(opt, net, dataloader)

def metric0(opt, net, dataloader):
	loss = util.AverageMeter()
	accuracy = util.AverageMeter()
	precision = util.AverageMeter()

	for i, (data, target) in enumerate(dataloader):
		if opt['cuda']:
			data, target = data.cuda(), target.cuda()

		data = torch.autograd.Variable(data)
		target = torch.autograd.Variable(target)

		logits = net(data, target)
		logits_data = logits.data.numpy()
		target_data = target.data.numpy()

		count = target_data.shape[0] * target_data.shape[1]

		l = np.mean(- target_data * np.log(logits_data + 1e-8) - (1 - target_data) * np.log(1 - logits_data + 1e-8))
		loss.update(l, n=count)

		p = logits_data * target_data
		p[p <= 0.5] = 0
		p = np.sum(p) / (np.sum(target_data) + 1e-8)

		precision.update(p)

		logits_data[logits_data > 0.5] = 1
		logits_data[logits_data <= 0.5] = -1
		target_data[target_data == 0.0] = -1

		a = np.sum(target_data * logits_data) / float(count)
		accuracy.update(a, n=count)

	logger.info('Validate: cross entropy is {} and accuracy is {} and precision is {}'.format(loss.avg, accuracy.avg, precision.avg))
	return loss.avg

def metric1(opt, net, dataloader):

	if opt['finetune']:
		loss = util.AverageMeter()
		precision = util.AverageMeter()
		recall = util.AverageMeter()

		industry_loss = util.AverageMeter()
		accuracy1 = util.AverageMeter()
		accuracy2 = util.AverageMeter()
		accuracy3 = util.AverageMeter()

		for i, (data, length, target, indices, mask) in enumerate(dataloader):
			data = data.squeeze(0)
			length = length.squeeze(0)
			target = target.squeeze(0)
			indices = indices.squeeze(0)
			mask = mask.squeeze(0)

			if opt['cuda']:
				data, target, indices, mask = data.cuda(async = True), target.cuda(async = True), indices.cuda(async = True), mask.cuda(async = True)

			data = torch.autograd.Variable(data)
			target = torch.autograd.Variable(target)
			length = torch.autograd.Variable(length)
			indices = torch.autograd.Variable(indices)
	 		mask = torch.autograd.Variable(mask)

			l, industry_l, predict, predict1, predict2, predict3 = net(data, length, target, indices, mask)

			predict = predict.cpu().data.numpy()
			indices = indices.cpu().data.numpy()
			mask = mask.cpu().data.numpy()

			predict1 = predict1.cpu().data.numpy()
			predict2 = predict2.cpu().data.numpy()
			predict3 = predict3.cpu().data.numpy()


			batch_size = predict.shape[0]
			loss.update(l.data[0], n=batch_size)
			industry_loss.update(industry_l.data[0], n=batch_size)
			
			threshold = 0.5
			predict[predict > 0.5] = 1
			predict[predict <= 0.5] = 0
			predict = predict * mask

			retrieved = np.sum(predict)
			relevant = np.sum(indices)
			intersect = np.sum(predict * indices)
			
			print retrieved, relevant, intersect

			p = intersect / (retrieved + 1e-8)
			r = intersect / (relevant + 1e-8)

			precision.update(p, n=retrieved)
			recall.update(r, n=relevant)

			target = target.cpu().data.numpy()
			predict1 = np.argmax(predict1, axis = 1)
			predict2 = np.argmax(predict2, axis = 1)
			predict3 = np.argmax(predict3, axis = 1)

			accuracy1.update(np.sum(predict1 == target[:, 0]) / float(batch_size), n=batch_size)
			accuracy2.update(np.sum(predict2 == target[:, 1]) / float(batch_size), n=batch_size)
			accuracy3.update(np.sum(predict3 == target[:, 2]) / float(batch_size), n=batch_size)

		f1 = 2 * precision.avg * recall.avg / (precision.avg + recall.avg + 1e-8)
			
		logger.info('Validate: loss: {:.3f}, industry loss: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}, accuracy1: {:.3f}, accuracy2: {:.3f}, accuracy3: {:.3f} '.format(loss.avg, industry_loss.avg, precision.avg, recall.avg, f1, accuracy1.avg, accuracy2.avg, accuracy3.avg))
		opt = net.module.opt
		return loss.avg * opt['lambda'] + industry_loss.avg * opt['alpha']

	else:
		loss = util.AverageMeter()
		accuracy1 = util.AverageMeter()
		accuracy2 = util.AverageMeter()
		accuracy3 = util.AverageMeter()

		for i, (data, length, target, indices, mask) in enumerate(dataloader):
			data = data.squeeze(0)
			length = length.squeeze(0)
			target = target.squeeze(0)
			indices = indices.squeeze(0)
			mask = mask.squeeze(0)

			if opt['cuda']:
				data, target, indices, mask = data.cuda(async = True), target.cuda(async = True), indices.cuda(async = True), mask.cuda(async = True)

			data = torch.autograd.Variable(data)
			target = torch.autograd.Variable(target)
			length = torch.autograd.Variable(length)
			indices = torch.autograd.Variable(indices)
	 		mask = torch.autograd.Variable(mask)

			l, (predict1, predict2, predict3) = net(data, length, target, indices, mask)

			predict1 = predict1.cpu().data.numpy()
			predict2 = predict2.cpu().data.numpy()
			predict3 = predict3.cpu().data.numpy()

			target = target.cpu().data.numpy()

			batch_size = target.shape[0]
			loss.update(l.data[0], n=batch_size)

			predict1 = np.argmax(predict1, axis = 1)
			predict2 = np.argmax(predict2, axis = 1)
			predict3 = np.argmax(predict3, axis = 1)

			accuracy1.update(np.sum(predict1 == target[:, 0]) / float(batch_size), n=batch_size)
			accuracy2.update(np.sum(predict2 == target[:, 1]) / float(batch_size), n=batch_size)
			accuracy3.update(np.sum(predict3 == target[:, 2]) / float(batch_size), n=batch_size)

		logger.info('Validate: loss: {:.3f}, accuracy1: {:.3f}, accuracy2: {:.3f}, accuracy3: {:.3f}'.format(loss.avg, accuracy1.avg, accuracy2.avg, accuracy3.avg))

		return loss.avg 

def metric2(opt, net, dataloader):
	return metric1(opt, net, dataloader)

def metric3(opt, net, dataloader):
	return metric1(opt, net, dataloader)

def metric4(opt, net, dataloader):
	return metric1(opt, net, dataloader)


