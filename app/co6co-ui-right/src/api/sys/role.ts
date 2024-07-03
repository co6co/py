//import request from '../../utils/request'
//import * as api_type from '../types'
const base_URL = '/api/role';
import { create_svc, create_association_svc } from '../base';
const services = create_svc(base_URL);

export default services;
const association_service = create_association_svc(base_URL);
export { association_service };
