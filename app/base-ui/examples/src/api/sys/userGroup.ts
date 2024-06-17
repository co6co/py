//import request from '../../utils/request'
//import * as api_type from '../types'
import { create_tree_svc, create_association_svc } from '../'

const base_URL = '/api/userGroup'
export default create_tree_svc(base_URL)
const association_service = create_association_svc(base_URL)
export { association_service }
