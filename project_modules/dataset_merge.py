
def data_merge(left_data,right_data):
    return left_data.merge(right_data,left_on='evidence_id',right_on='doc_id',how='left')