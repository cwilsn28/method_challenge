
TO_MERGE = {
    'PreferredLoginDevice': {
        'Phone': 'Mobile Phone',
    },
    'PreferredPaymentMode': {
        'Cash on Delivery': 'COD',
        'CC': 'Credit Card',
    },
}

TO_ENCODE_OHE = [
    'MaritalStatus',
    'PreferredLoginDevice',
    'PreferredPaymentMode',
]

TO_ENCODE_COUNT = [
    'Gender',
    'PreferedOrderCat',
]

# Reduce dimensionality by training on a subset of relevant features
FEATURES = [
    'Gender',
    'Tenure',
    'CouponUsed',
    'CashbackAmount',
    'DaySinceLastOrder',
    'OrderCount',
    'OrderAmountHikeFromlastYear',
    'SatisfactionScore',
    'HourSpendOnApp',
]