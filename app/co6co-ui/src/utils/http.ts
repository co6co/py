import { HttpContentType } from '@/constants';

/**
 * http 请求的 content-type 类型
 * @param contentType header 中的 content-type
 * @returns
 */
export const getContypeType = (contentType: string): HttpContentType => {
	contentType = contentType || '';
	if (contentType.includes('application/json')) {
		return HttpContentType.json;
	} else if (contentType.includes('text/html')) {
		//new DOMParser().parseFromString(str, 'text/html'))
		return HttpContentType.html;
	} else if (
		contentType.includes('application/xml') ||
		contentType.includes('text/xml')
	) {
		//new DOMParser().parseFromString(str, 'application/xml')
		return HttpContentType.xml;
	} else if (contentType.includes('image/')) {
		//const link = document.createElement('a');
		//link.href = URL.createObjectURL(blob);
		//link.download = 'downloaded-file.ext';
		//link.click();
		//return link;
		return HttpContentType.image;
	} else if (contentType.includes('video/')) {
		//const link = document.createElement('a');
		//link.href = URL.createObjectURL(blob);
		//link.download = 'downloaded-file.ext';
		//link.click();
		//return link;
		return HttpContentType.video;
	} else if (contentType.includes('application/octet-stream'))
		return HttpContentType.stream;
	else return HttpContentType.text;
};
