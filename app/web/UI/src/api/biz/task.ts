const base_URL = '/api/biz/sys/task'
import { create_svc } from 'co6co-right'
const { exist_svc, add_svc, edit_svc, del_svc, get_table_svc } = create_svc(base_URL)
export { exist_svc, add_svc, edit_svc, del_svc, get_table_svc }
