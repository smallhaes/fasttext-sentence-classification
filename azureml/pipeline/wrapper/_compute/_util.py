# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
from pkg_resources import resource_string


cmakscompute_payload_template = json.loads(
    resource_string(__name__, 'data/cmakscompute_template.json').decode('ascii'))
