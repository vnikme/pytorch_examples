import json
import base64
import xml.etree.ElementTree as ET
import sys
import glob
import torch
from model import Seq2SeqVAE
import random
import numpy as np
import time
from html.parser import HTMLParser
import sentencepiece as spm


def text_to_codes(item, sp):
    return sp.EncodeAsIds(item) if item else None


def load_raw_data(path, sp):
    data = []
    for line in open(path, 'rt'):
        item = json.loads(line)
        data.append([text_to_codes(item['en'], sp), text_to_codes(item['ru'], sp)])
    return data


def prepare_batch(items, sp, model, device):
    zero, start, stop = sp.PieceToId('<unk>'), sp.PieceToId('<s>'), sp.PieceToId('</s>')
    max_batch_length = max(map(lambda x: max(len(x[0]) if x[0] else 0, len(x[1]) if x[1] else 0), items))
    encoder0_input, encoder1_input, decoder0_input, decoder1_input, target0_output, target1_output, lengths0, lengths1 = [], [], [], [], [], [], [], []
    for item in items:
        if item[0] and item[1]:
            encoder0_input.append(item[0] + [zero] * (max_batch_length - len(item[0])))
            encoder0_input.append(item[1] + [zero] * (max_batch_length - len(item[1])))
            encoder1_input.append(item[0] + [zero] * (max_batch_length - len(item[0])))
            encoder1_input.append(item[1] + [zero] * (max_batch_length - len(item[1])))
            decoder0_input.append([start] + item[0] + [zero] * (max_batch_length - len(item[0])))
            decoder0_input.append([start] + item[0] + [zero] * (max_batch_length - len(item[0])))
            decoder1_input.append([start] + item[1] + [zero] * (max_batch_length - len(item[1])))
            decoder1_input.append([start] + item[1] + [zero] * (max_batch_length - len(item[1])))
            target0_output.append(item[0] + [stop] * (max_batch_length - len(item[0]) + 1))
            target0_output.append(item[0] + [stop] * (max_batch_length - len(item[0]) + 1))
            target1_output.append(item[1] + [stop] * (max_batch_length - len(item[1]) + 1))
            target1_output.append(item[1] + [stop] * (max_batch_length - len(item[1]) + 1))
            lengths0.append(len(item[0]))
            lengths0.append(len(item[1]))
            lengths1.append(len(item[0]))
            lengths1.append(len(item[1]))
        elif item[0]:
            encoder0_input.append(item[0] + [zero] * (max_batch_length - len(item[0])))
            decoder0_input.append([start] + item[0] + [zero] * (max_batch_length - len(item[0])))
            target0_output.append(item[0] + [stop] * (max_batch_length - len(item[0]) + 1))
            lengths0.append(len(item[0]))
        elif item[1]:
            encoder1_input.append(item[1] + [zero] * (max_batch_length - len(item[1])))
            decoder1_input.append([start] + item[1] + [zero] * (max_batch_length - len(item[1])))
            target1_output.append(item[1] + [stop] * (max_batch_length - len(item[1]) + 1))
            lengths1.append(len(item[1]))
    encoder0_input = torch.tensor(encoder0_input, dtype=torch.long).to(device)
    encoder0_input = model.embed(encoder0_input)
    encoder1_input = torch.tensor(encoder1_input, dtype=torch.long).to(device)
    encoder1_input = model.embed(encoder1_input)
    decoder0_input = torch.tensor(decoder0_input, dtype=torch.long).to(device)
    decoder0_input = model.embed(decoder0_input)
    decoder1_input = torch.tensor(decoder1_input, dtype=torch.long).to(device)
    decoder1_input = model.embed(decoder1_input)
    target0_output = torch.tensor(target0_output, dtype=torch.long).to(device)
    target1_output = torch.tensor(target1_output, dtype=torch.long).to(device)
    lengths0 = torch.tensor(lengths0, dtype=torch.long).to(device)
    lengths1 = torch.tensor(lengths1, dtype=torch.long).to(device)
    return encoder0_input, encoder1_input, decoder0_input, decoder1_input, target0_output, target1_output, lengths0, lengths1


def prepare_encoder_batch(items, sp, model, device):
    zero, start, stop = sp.PieceToId('<unk>'), sp.PieceToId('<s>'), sp.PieceToId('</s>')
    max_batch_length = max(map(len, items))
    encoder_input, lengths = [], []
    for item in items:
        encoder_input.append(item + [zero] * (max_batch_length - len(item)))
        lengths.append(len(item))
    encoder_input = torch.tensor(encoder_input, dtype=torch.long).to(device)
    encoder_input = model.embed(encoder_input)
    lengths = torch.tensor(lengths, dtype=torch.long).to(device)
    return encoder_input, lengths


def iterate_data(data, model, batch_size, sp, max_length, device):
    items, max_batch_length = [], 0
    idxs = list(range(len(data)))
    random.shuffle(idxs)
    #idxs = idxs[:1000]
    percentile = max(int(len(idxs) / 100), 1)
    for i in range(len(idxs)):
        if i % percentile == 0:
            print('{}/{} ({}%)'.format(i, len(idxs), i * 100.0 / len(idxs)), file=sys.stderr)
            sys.stderr.flush()
        idx = idxs[i]
        item = data[idx]
        item = [item[0][:max_length] if item[0] else item[0], item[1][:max_length] if item[1] else item[1]]
        if (len(items) + 1) * max(max_batch_length, len(item[0]) if item[0] else 0, len(item[1]) if item[1] else 0) > batch_size:
            if items:
                yield prepare_batch(items, sp, model, device)
            items, max_batch_length = [], 0
        items.append(item)
        max_batch_length = max(max_batch_length, len(item[0]) if item[0] else 0, len(item[1]) if item[1] else 0)
    if items:
        yield prepare_batch(items, sp, model, device)


def generate(model, idx, sp, max_length, prime, state, device):
    stop = sp.PieceToId('</s>')
    prime = sp.EncodeAsIds(prime)
    result = prime
    prime = [sp.PieceToId('<s>')] + prime
    while len(result) <= max_length:
        decoder_input = torch.tensor([prime], dtype=torch.long).to(device)
        decoder_input = model.embed(decoder_input)
        output, state = model.decode(decoder_input, state, idx)
        output = torch.exp(output[0][-1])
        output = list(output)
        max_idxs = list(range(len(output)))
        max_idxs = sorted(max_idxs, key = lambda x: -output[x])[:3]
        output = [float(output[x]) for x in max_idxs]
        s = sum(output)
        for i in range(len(output)):
            output[i] /= s
        c = int(np.random.choice(max_idxs, p=output))
        if c == stop:
            break
        result.append(c)
        prime = [c]
    return sp.DecodeIds(result)


def check_on_manual_examples(model, device, sp):
    items = [
        'Обращение Президента Российской Федерации Владимира Путина к Парламенту',
        'Обращение Президента Российской Федерации Дмитрия Медведева к Парламенту',
        'Владимир Путин провел рабочую встречу с представителями бизнеса',
        'Дмитрий Медведев провел рабочую встречу с представителями бизнеса',
        'Владимир Путин провёл встречу с руководителями профсоюзов',
        'Дмитрий Медведев провёл встречу с руководителями профсоюзов',
        'Владимир Путин сложил полномочия Президента Российской Федерации',
        'Дмитрий Медведев сложил полномочия Президента Российской Федерации',
    ]
    items = list(map(lambda item: sp.EncodeAsIds(item), items))
    encoder_input, lengths = prepare_encoder_batch(items, sp, model, device)
    h = torch.zeros([model.n_layers, lengths.shape[0], model.hidden_size]).to(device)
    mu, logvar = model.encode(encoder_input, lengths, h)
    d1 = [float(mu[1][i] - mu[0][i]) for i in range(mu.shape[1])]
    d2 = [float(mu[3][i] - mu[2][i]) for i in range(mu.shape[1])]
    d3 = [float(mu[7][i] - mu[0][i]) for i in range(mu.shape[1])]
    print(sum([(d1[i] - d2[i]) * (d1[i] - d2[i]) for i in range(len(d2))]), file=sys.stderr)
    print(sum([(d3[i] - d2[i]) * (d3[i] - d2[i]) for i in range(len(d2))]), file=sys.stderr)
    for i in range(len(items)):
        print(sp.DecodeIds(items[i]), file=sys.stderr)
        print(generate(model, 0, sp, 300, '', model.latent_2_hidden(mu[i:i+1,:]), device), file=sys.stderr)
        print(generate(model, 1, sp, 300, '', model.latent_2_hidden(mu[i:i+1,:]), device), file=sys.stderr)
        print('', file=sys.stderr)


def main(data_path, models_path, start_epoch, learning_rate, prev_loss):
    sp = spm.SentencePieceProcessor()
    sp.Load("bpe1k.model")
    dict_size = sp.GetPieceSize()

    cuda = torch.cuda.is_available() #and False
    device = torch.device("cuda" if cuda else "cpu")
    cpu_device = torch.device("cpu")
    print(cuda, device)

    embed_size, hidden_size, n_layers, latent_size, dropout = 16, 256, 4, 64, 0.2
    batch_size, clip, max_length = 10000, 0.1, 255
    vae_freq = 100

    print(embed_size, hidden_size, n_layers, latent_size, dropout, batch_size, clip, max_length, vae_freq)

    model = Seq2SeqVAE(sp, embed_size, hidden_size, n_layers, latent_size, dropout, device)
    if start_epoch >= 0:
        model.load_state_dict(torch.load('{}_{}'.format(models_path, start_epoch), map_location=device))
        model.to(device)

    if False:
        model.to(cpu_device)
        model.eval()
        for _ in range(10):
            h = torch.randn(1, latent_size).to(cpu_device)
            h = model.latent_2_hidden(h)
            print(generate(model, 0, sp, 300, '', h, cpu_device), file=sys.stderr)
            print(generate(model, 1, sp, 300, '', h, cpu_device), file=sys.stderr)
            print('', file=sys.stderr)
        check_on_manual_examples(model, cpu_device, sp)
        return

    data = load_raw_data(data_path, sp)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch = start_epoch
    while True:
        epoch += 1
        start_time = time.time()
        total_loss, total_recon_loss, total_kld_loss = 0.0, 0.0, 0.0
        total_count = 0
        step, lr_changed = 0, False
        for encoder0_input, encoder1_input, decoder0_input, decoder1_input, target0_output, target1_output, lengths0, lengths1 in iterate_data(data, model, batch_size, sp, max_length, device):
            step += 1
            model.to(device)
            model.train()
            model.zero_grad()
            h0 = torch.zeros([n_layers, lengths0.shape[0], hidden_size]).to(device)
            mu0, logvar0 = model.encode(encoder0_input, lengths0, h0)
            if vae_freq and step % vae_freq == 0:
                z0 = model.reparameterize(mu0, logvar0)
            else:
                z0 = mu0
            z0 = model.latent_2_hidden(z0)
            h1 = torch.zeros([n_layers, lengths1.shape[0], hidden_size]).to(device)
            mu1, logvar1 = model.encode(encoder1_input, lengths1, h1)
            if vae_freq and step % vae_freq == 0:
                z1 = model.reparameterize(mu1, logvar1)
            else:
                z1 = mu1
            z1 = model.latent_2_hidden(z1)
            output0, _ = model.decode(decoder0_input, z0, 0)
            output1, _ = model.decode(decoder1_input, z1, 1)
            recon_loss_0 = criterion(output0.view([-1, dict_size]), target0_output.view(-1))
            recon_loss_1 = criterion(output1.view([-1, dict_size]), target1_output.view(-1))
            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            if vae_freq > 0 and step % vae_freq == 0:
                kld_loss_0 = (-0.5 * torch.sum(1 + logvar0 - mu0.pow(2) - logvar0.exp()))
                kld_loss_1 = (-0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp()))
            else:
                kld_loss_0, kld_loss_1 = 0.0, 0.0
            recon_loss = recon_loss_0 + recon_loss_1
            kld_loss = kld_loss_0 + kld_loss_1
            loss_0 = float(recon_loss_0) + 0.001 * float(kld_loss_0)
            loss_1 = float(recon_loss_1) + 0.001 * float(kld_loss_1)
            loss = recon_loss + 0.001 * kld_loss
            total_loss += (loss_0 * lengths0.shape[0] + loss_1 * lengths1.shape[0])
            total_recon_loss += (float(recon_loss_0) * lengths0.shape[0] + float(recon_loss_1) * lengths1.shape[0])
            total_kld_loss += (float(kld_loss_0) * lengths0.shape[0] + float(kld_loss_1) * lengths1.shape[0])
            total_count += (lengths0.shape[0] + lengths1.shape[0])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            if total_count > 10000 and total_loss / total_count > prev_loss * 1.1:
                learning_rate /= 2
                epoch -= 1
                if epoch == -1:
                    model = Seq2SeqVAE(sp, embed_size, hidden_size, n_layers, latent_size, dropout, device)
                else:
                    model.load_state_dict(torch.load('{}_{}'.format(models_path, epoch)))
                opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
                print('Epoch {}, loss {}, recon_loss {}, kld_loss {}, time {}'.format(epoch, total_loss / total_count, total_recon_loss / total_count, total_kld_loss / total_count, time.time() - start_time))
                print('New learning rate is {}, restarting epoch'.format(learning_rate))
                sys.stdout.flush()
                lr_changed = True
                break
            if step % 5000 == 0:
                print('loss {}, recon_loss {}, kld_loss {}, time {}'.format(total_loss / total_count, total_recon_loss / total_count, total_kld_loss / total_count, time.time() - start_time), file=sys.stderr)
                model.to(cpu_device)
                model.eval()
                h = torch.randn(1, latent_size).to(cpu_device)
                h = model.latent_2_hidden(h)
                print(generate(model, 0, sp, 300, '', h, cpu_device), file=sys.stderr)
                print(generate(model, 1, sp, 300, '', h, cpu_device), file=sys.stderr)
                check_on_manual_examples(model, cpu_device, sp)
                sys.stderr.flush()
                model.to(device)
        if lr_changed:
            continue
        print('Epoch {}, loss {}, recon_loss {}, kld_loss {}, time {}'.format(epoch, total_loss / total_count, total_recon_loss / total_count, total_kld_loss / total_count, time.time() - start_time))
        sys.stdout.flush()
        prev_loss = total_loss / total_count
        torch.save(model.state_dict(), '{}_{}'.format(models_path, epoch))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))

