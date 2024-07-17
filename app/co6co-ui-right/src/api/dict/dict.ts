const base_URL = '/api/dict';

import { create_svc } from '../base';

const { add_svc, edit_svc, del_svc, get_table_svc, exist_svc } =
	create_svc(base_URL);
export { add_svc, edit_svc, del_svc, get_table_svc, exist_svc };
