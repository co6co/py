import { create_tree_svc } from 'co6co-right'
const base_URL = '/api/view'

const _service = create_tree_svc(base_URL)
const view_route_svc = _service.get_select_tree_svc
export { view_route_svc }
