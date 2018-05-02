void Get_Descriptor(const cv::KeyPoint& kpt, uint8_t * __restrict dst) const
{
    assert(kpt.angle == kpt.angle); // check is not NaN
    
    // Get the information from the keypoint
    float pixelSamplingRatio = 1.0f / (float)(1 << kpt.octave);
    
    float xf = kpt.pt.x * pixelSamplingRatio;
    float yf = kpt.pt.y * pixelSamplingRatio;
    cv::Point cornerPoint(cvRound(xf), cvRound(yf));
    
    float level = kpt.class_id;
    const cv::Mat_<float>& img = evolution_[level].Lt;
    
    int histlen = (HSG_DESCR_WIDTH_PADDED)*(HSG_DESCR_WIDTH_PADDED)*(HSG_DESCR_HIST_BINS_PADDED);
    cv::AutoBuffer<float> buf(histlen);
    float *hist = buf;
    memset (hist, 0, histlen*sizeof(float));
    
    calculateGradientHistogram(kpt, img, cornerPoint, hist);
    finalizeHistogram(dst, hist);
}

void calculateGradientHistogram(const cv::KeyPoint& kpt, const cv::Mat_<float>& img, const cv::Point& pt, float *hist) const{
    float exp_scale = -1.f / (HSG_DESCR_WIDTH * HSG_DESCR_WIDTH * 0.5f);
    float bins_per_rad = HSG_DESCR_HIST_BINS / (2.0f * CV_PI);
    
    const int rows = img.rows, cols = img.cols;
    
    const float scl = kpt.size*0.5f;
    float hist_scale = HSG_DESCR_SCL_FCTR * scl;
    int radius = cvRound( 1.4142135623730951f * (HSG_DESCR_WIDTH + 1)*hist_scale * 0.5f);
    // Clip the radius to the diagonal of the image to avoid autobuffer too large exception
    radius = std::min(radius, (int)sqrt((double)cols*cols + rows*rows));
    
    //rotation agngle towards main orientation
    float cos_t = std::cos(-kpt.angle);
    float sin_t = std::sin(-kpt.angle);
    cos_t /= hist_scale;
    sin_t /= hist_scale;
    
 
    for (int i = -radius; i <= radius; i++){
        for (int j = -radius; j <= radius; j++)
        {
            float c_rot = j * cos_t - i * sin_t;
            float r_rot = j * sin_t + i * cos_t;
            float rbin = r_rot + HSG_DESCR_WIDTH / 2 - 0.5f;
            float cbin = c_rot + HSG_DESCR_WIDTH / 2 - 0.5f;
            
            if (rbin > -1 && rbin < HSG_DESCR_WIDTH && cbin > -1 && cbin < HSG_DESCR_WIDTH){
                
                int r = pt.y + i, c = pt.x + j;
                
                if(r > 0 && r < rows - 1 && c > 0 && c < cols - 1)
                {
                    float dx = (img(r, c + 1) - img(r, c - 1));
                    float dy = (img(r - 1, c) - img(r + 1, c));
                    float orientation = deg2rad(cv::fastAtan2(dy, dx));
                    float obin = (kpt.angle + orientation - CV_PI * 2)*bins_per_rad;
                    
                    float mag = std::sqrt(dx * dx + dy * dy) * std::exp((c_rot * c_rot + r_rot * r_rot)*exp_scale);
                    //vision::fastsqrt1(dx * dx + dy * dy) * vision::fastexp6((c_rot * c_rot + r_rot * r_rot)*exp_scale);
                    
                    
                    int32_t r0 = std::floor(rbin);
                    int32_t c0 = std::floor(cbin);
                    int32_t o0 = std::floor(obin);
                    
                    float rbinPartial = rbin - r0;
                    float cbinPartial = cbin - c0;
                    float obinPartial = obin - o0;
                    
                    if (o0 < 0)
                        o0 += HSG_DESCR_HIST_BINS;
                    if (o0 >= HSG_DESCR_HIST_BINS)
                        o0 -= HSG_DESCR_HIST_BINS;
                    
                    
                    int idx = getHistIndex(r0,c0, o0);
                    triLinearInterpolation(mag, rbinPartial, cbinPartial, obinPartial, &(hist[idx]));
                }
            }
        }
    }
}

void triLinearInterpolation(const float mag, const float rbin, const float cbin, const float obin, float *histogram) const {
    // histogram update using tri-linear interpolation
    float v_rco111 = rbin     * cbin     * obin     * mag;
    float v_rco110 = rbin     * cbin     * (1-obin) * mag;
    float v_rco101 = rbin     * (1-cbin) * obin     * mag;
    float v_rco100 = rbin     * (1-cbin) * (1-obin) * mag;
    float v_rco011 = (1-rbin) * cbin     * obin     * mag;
    float v_rco010 = (1-rbin) * cbin     * (1-obin) * mag;
    float v_rco001 = (1-rbin) * (1-cbin) * obin     * mag;
    float v_rco000 = (1-rbin) * (1-cbin) * (1-obin) * mag;
    
    int nBinBlock = HSG_DESCR_WIDTH_PADDED, nBins = HSG_DESCR_HIST_BINS_PADDED;
    
    histogram[0] += v_rco000;
    histogram[1] += v_rco001;
    histogram[nBins] += v_rco010;
    histogram[nBins + 1] += v_rco011;
    histogram[nBinBlock*nBins] += v_rco100;
    histogram[nBinBlock*nBins + 1] += v_rco101;
    histogram[(nBinBlock + 1)*nBins] += v_rco110;
    histogram[(nBinBlock + 1)*nBins + 1] += v_rco111;
}

void triLogLinearInterpolation(const float mag, const float rbin, const float cbin, const float obin, float *histogram) const {
    // histogram update using tri-linear interpolation
    float log_r = std::log(1 + rbin);
    float log_c = std::log(1 + cbin);
    float log_o = std::log(1 + obin);
    float log_r_1 = std::log(1 + 1 - rbin);
    float log_c_1 = std::log(1 + 1 - cbin);
    float log_o_1 = std::log(1 + 1 - obin);
    
    float log_mag = log(1+ mag);
    
    float v_rco111 = log_r   + log_c   + log_o   + log_mag;
    float v_rco110 = log_r   + log_c   + log_o_1 + log_mag;
    float v_rco101 = log_r   + log_c_1 + log_o   + log_mag;
    float v_rco100 = log_r   + log_c_1 + log_o_1 + log_mag;
    float v_rco011 = log_r_1 + log_c   + log_o   + log_mag;
    float v_rco010 = log_r_1 + log_c   + log_o_1 + log_mag;
    float v_rco001 = log_r_1 + log_c_1 + log_o   + log_mag;
    float v_rco000 = log_r_1 + log_c_1 + log_o_1 + log_mag;
    
    int nBinBlock = HSG_DESCR_WIDTH_PADDED, nBins = HSG_DESCR_HIST_BINS_PADDED;
    
    histogram[0] += v_rco000;
    histogram[1] += v_rco001;
    histogram[nBins] += v_rco010;
    histogram[nBins + 1] += v_rco011;
    histogram[nBinBlock*nBins] += v_rco100;
    histogram[nBinBlock*nBins + 1] += v_rco101;
    histogram[(nBinBlock + 1)*nBins] += v_rco110;
    histogram[(nBinBlock + 1)*nBins + 1] += v_rco111;
}
void AKAZE::triExponentialLinearInterpolation(const float mag, const float rbin, const float cbin, const float obin, float *histogram) const {
    // histogram update using tri-linear interpolation
    float v_rco111 = vision::fastexp6(rbin      + cbin      + obin    )*mag;
    float v_rco110 = vision::fastexp6(rbin      + cbin      + 1 - obin)*mag;
    float v_rco101 = vision::fastexp6(rbin      + 1 - cbin  + obin    )*mag;
    float v_rco100 = vision::fastexp6(rbin      + 1 - cbin  + 1 - obin)*mag;
    float v_rco011 = vision::fastexp6(1 - rbin  + cbin      + obin    )*mag;
    float v_rco010 = vision::fastexp6(1 - rbin  + cbin      + 1 - obin)*mag;
    float v_rco001 = vision::fastexp6(1 - rbin  + 1 - cbin  + obin    )*mag;
    float v_rco000 = vision::fastexp6(1 - rbin  + 1 - cbin  + 1 - obin)*mag;
    
    int nBinBlock = HSG_DESCR_WIDTH_PADDED, nBins = HSG_DESCR_HIST_BINS_PADDED;
    
    histogram[0] += v_rco000;
    histogram[1] += v_rco001;
    histogram[nBins] += v_rco010;
    histogram[nBins + 1] += v_rco011;
    histogram[nBinBlock*nBins] += v_rco100;
    histogram[nBinBlock*nBins + 1] += v_rco101;
    histogram[(nBinBlock + 1)*nBins] += v_rco110;
    histogram[(nBinBlock + 1)*nBins + 1] += v_rco111;
}
void triSquareLinearInterpolation(const float mag, const float rbin, const float cbin, const float obin, float *histogram) const {
    // histogram update using tri-linear interpolation
    float sq_r = rbin*rbin;
    float sq_c = cbin*cbin;
    float sq_o = obin*obin;
    float sq_r_1 = (1-rbin)*(1-rbin);
    float sq_c_1 = (1-cbin)*(1-cbin);
    float sq_o_1 = (1-obin)*(1-obin);
    
    float v_rco111 = sq_r   * sq_c   * sq_o   * mag;
    float v_rco110 = sq_r   * sq_c   * sq_o_1 * mag;
    float v_rco101 = sq_r   * sq_c_1 * sq_o   * mag;
    float v_rco100 = sq_r   * sq_c_1 * sq_o_1 * mag;
    float v_rco011 = sq_r_1 * sq_c   * sq_o   * mag;
    float v_rco010 = sq_r_1 * sq_c   * sq_o_1 * mag;
    float v_rco001 = sq_r_1 * sq_c_1 * sq_o   * mag;
    float v_rco000 = sq_r_1 * sq_c_1 * sq_o_1 * mag;
    
    int nBinBlock = HSG_DESCR_WIDTH_PADDED, nBins = HSG_DESCR_HIST_BINS_PADDED;
    
    histogram[0] += v_rco000;
    histogram[1] += v_rco001;
    histogram[nBins] += v_rco010;
    histogram[nBins + 1] += v_rco011;
    histogram[nBinBlock*nBins] += v_rco100;
    histogram[nBinBlock*nBins + 1] += v_rco101;
    histogram[(nBinBlock + 1)*nBins] += v_rco110;
    histogram[(nBinBlock + 1)*nBins + 1] += v_rco111;
}


void finalizeHistogram(uint8_t * __restrict dst, float *hist) const{
    cv::AutoBuffer<float> buf(HSG_DESCR_LEN);
    float *desc = buf;
    memset (desc, 0, HSG_DESCR_LEN*sizeof(float));
    
    // finalize histogram, since the orientation histograms are circular
    for (int x = 0; x < HSG_DESCR_WIDTH; x++){
        for (int y = 0; y < HSG_DESCR_WIDTH; y++)
        {
            int index = getHistIndex(x,y);
            hist[index] += hist[index + HSG_DESCR_HIST_BINS];
            hist[index + 1] += hist[index + HSG_DESCR_HIST_BINS + 1];
            for (int z = 0; z < HSG_DESCR_HIST_BINS; z++){
                desc[(x*HSG_DESCR_WIDTH + y)*HSG_DESCR_HIST_BINS + z] = hist[index + z];
            }
        }
    }
    
    float nrm2 = std::accumulate (desc, desc+HSG_DESCR_LEN, 0.0, [](float x, float y) {return x+y*y;});
    float thr = std::sqrt(nrm2)*HSG_DESCR_MAG_THR;
    
    std::replace_if(desc, desc+HSG_DESCR_LEN, [thr](float n) { return n > thr; }, thr);
    nrm2 = std::accumulate (desc, desc+HSG_DESCR_LEN, 0.0, [](float x, float y) {return x+y*y;});
    
    nrm2 = HSG_INT_DESCR_FCTR / std::max(std::sqrt(nrm2), FLT_EPSILON);
    
    std::transform(desc, desc+HSG_DESCR_LEN, dst, [nrm2](float x) {return cv::saturate_cast<uchar>(x*nrm2);});
}

