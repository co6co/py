const base_URL = '/api/config';

import { create_svc } from '../base';

const { exist_svc, add_svc, edit_svc, del_svc, get_table_svc } =
	create_svc(base_URL);
export { exist_svc, add_svc, edit_svc, del_svc, get_table_svc };
