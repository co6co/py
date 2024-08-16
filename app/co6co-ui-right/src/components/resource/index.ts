import { withInstall } from 'co6co';

import htmlPlayer from './src/htmlPlayer';

export const HtmlPlayer = withInstall(htmlPlayer);
export default HtmlPlayer;
import imageView from './src/imageView';
export const ImageView = withInstall(imageView);
import imageListView from './src/imageListView';
export const ImageListView = withInstall(imageListView);

import imgVideo from './src/imgVideo';
export const ImgVideo = withInstall(imgVideo);
export * from './src';
