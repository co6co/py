import axios  from '../utils/request';
export const fetchData = () => {
    return axios ({
        url: './demo/table.json',
        method: 'get'
    });
};  