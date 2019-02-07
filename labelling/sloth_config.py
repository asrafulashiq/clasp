from PyQt4.QtGui import QPen
from PyQt4.Qt import Qt
from sloth.items import RectItem


class CustomRectItem(RectItem):
    def __init__(self, *args, **kwargs):
        RectItem.__init__(self, *args, **kwargs)

        # set drawing pen to red with width 2
        self.setPen(QPen(Qt.red, 2))


LABELS = (
    {
        'attributes': {
            'class':      'Person',
        },
        'inserter': 'sloth.items.RectItemInserter',
        'item':     CustomRectItem,
        'hotkey':   'p',
        'text':     'Person',
    },
    {
        'attributes': {
            'class':      'binEMPTY',
        },
        'inserter': 'sloth.items.RectItemInserter',
        'item':     CustomRectItem,
        'hotkey':   'b',
        'text':     'binEMPTY',
    },
    {
        'attributes': {
            'class':      'binFULL',
        },
        'inserter': 'sloth.items.RectItemInserter',
        'item':     CustomRectItem,
        'hotkey':   'c',
        'text':     'binFULL',
    }

)
