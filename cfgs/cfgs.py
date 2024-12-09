def parse_common_args(parser):
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--backbone', type=str, default='ViT-B-16')
    parser.add_argument('--device_id', type=str, default='1')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_instance', type=int, default=4)
    parser.add_argument('--data_path', type=str, default='/data/ReidShareDataset', help="dataset path")
    parser.add_argument('--log_path', type=str, default='logs', help="log path")
    return parser


def parse_train_args(parser, datasets=None, classes=None, combine_all=False):
    if classes is None:
        classes = [1502, 11934, 1816, 1467]
    if datasets is None:
        datasets = ["Market", "cuhk_sysu", "CUHK02", "cuhk03"]
    parser = parse_common_args(parser)
    parser.add_argument('--prompt-epoch', type=int, default=120, help='prompt epoch')
    parser.add_argument('--prompt-domain-epoch', type=int, default=30, help='prompt domain epoch')
    parser.add_argument('--prior-epoch', type=int, default=3, help='prior epoch')
    parser.add_argument('--image-encoder-epoch', type=int, default=60, help='image encoder epoch')
    parser.add_argument('--prompt-lr', default=0.00035, type=float)
    parser.add_argument('--image-encoder-lr', default=0.000005, type=float)
    parser.add_argument('--beta', default=0.1, type=float, help='weight for apn-loss')
    parser.add_argument('--omega', default=0.01, type=float, help='weight for domain loss')
    parser.add_argument('--epsilon', default=0.1, type=float, help='weights for gradient backpropagation')
    parser.add_argument('--lamda', default=0.01, type=float, help='learning rate for domain classifier')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--size_train', default=[224, 224], type=list)
    parser.add_argument('--gray_scale', default=0.1, type=float)
    parser.add_argument('--pad', default=10, type=int)
    parser.add_argument('--random_horizontal_flip', default=0.4, type=float)
    parser.add_argument('--color_jitter', default=0.1, type=float)
    parser.add_argument('--aug_mix', default=0.6, type=float)
    parser.add_argument('--random_erasing', default=0.0, type=float)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--checkpoint_period', type=int, default=60)
    parser.add_argument('--log_period', type=int, default=100)
    parser.add_argument('--eval_period', type=int, default=10)
    parser.add_argument('--train_datasets', type=list, default=datasets)
    parser.add_argument('--classes', type=list, default=classes)
    parser.add_argument('--sampling_method', type=str, default='PK')
    parser.add_argument('--combine_all', type=bool, default=combine_all)
    return parser


def parse_test_args(parser, datasets=None):
    if datasets is None:
        datasets = ["PRID", "GRID", "VIPeR", "iLIDS"]
    parser = parse_common_args(parser)
    parser.add_argument('--size_test', default=[224, 224], type=list)
    parser.add_argument('--test_datasets', type=list, default=datasets)
    parser.add_argument('--test_batch_size', type=int, default=256)
    return parser


def protocol_1(parser,parsertest):
    return parse_train_args(parser, ["Market", "cuhk_sysu", "CUHK02", "cuhk03"], [1502, 11934, 1816, 1467],True), \
           parse_test_args(parsertest, ["PRID", "GRID", "VIPeR", "iLIDS"]),\
           'protocol_1'


def protocol_2_C3(parser,parsertest):
    return parse_train_args(parser, ["Market", "MSMT17", "cuhk_sysu"], [1502, 4101, 11934], False), \
           parse_test_args(parsertest, ["cuhk03"]),\
           'protocol_2_C3'


def protocol_3_C3(parser,parsertest):
    return parse_train_args(parser, ["Market", "MSMT17", "cuhk_sysu"], [1502, 4101, 11934], True), \
           parse_test_args(parsertest, ["cuhk03"]),\
           'protocol_2_C3'


def protocol_2_MS(parser,parsertest):
    return parse_train_args(parser, ["cuhk03", "cuhk_sysu", "Market"], [1467, 11934, 1502], False), \
           parse_test_args(parsertest, ["MSMT17"]),\
           'protocol_2_MS'


def protocol_3_MS(parser,parsertest):
    return parse_train_args(parser, ["cuhk03", "cuhk_sysu", "Market"], [1467, 11934, 1502], True), \
           parse_test_args(parsertest, ["MSMT17"]),\
           'protocol_3_MS'



def protocol_2_M(parser,parsertest):
    return parse_train_args(parser, ["cuhk03", "cuhk_sysu", "MSMT17"], [1467,11934,4101], False), \
           parse_test_args(parsertest, ["Market"]),\
           'protocol_2_M'


def protocol_3_M(parser,parsertest):
    return parse_train_args(parser, ["cuhk03", "cuhk_sysu", "MSMT17"], [1467,11934,4101], True), \
           parse_test_args(parsertest, ["Market"]),\
           'protocol_3_M'
