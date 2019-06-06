#include "header.hpp"

using namespace pcl;

float
calc_reachability(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const int &idx_base, const int &idx_nn, const std::vector<float> &knn_dist){
    float k_dist = knn_dist[idx_base];
    pcl::PointXYZ point1 = cloud->points[idx_base];
    pcl::PointXYZ point2 = cloud->points[idx_nn]; 
    float nn_dist = std::sqrt(point1.x*point2.x + point1.y*point2.y + point1.z*point2.z);

    return nn_dist;
}

void
inflo(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const float threshold, pcl::PointIndices::Ptr inliers, pcl::PointIndices::Ptr outliers, int _K){
//inflo(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float threshold){    
    int n_points = cloud->points.size();

    std::vector<std::vector<int>> knn_vec;
    std::vector<float> knn_dist;    
    std::vector<std::vector<int>> rnn_vec;
    std::vector<std::vector<int>> influential_nn_vec;


    for(int i=0;i<n_points;i++){
        std::vector<int> _knn_indices;
        std::vector<int> _rnn_indices;
        std::vector<int> _influential_nn_indices;        

        knn_vec.push_back(_knn_indices);
        knn_dist.push_back(0);
        rnn_vec.push_back(_rnn_indices);
        influential_nn_vec.push_back(_influential_nn_indices);
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud (cloud);

    int K = _K;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);

    int threads_ = 8;

    //pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> > (new pcl::search::KdTree<pcl::PointXYZ>);    
    std::chrono::system_clock::time_point  start, end; // 型は auto で可
    start = std::chrono::system_clock::now(); // 計測開始時間
    #pragma omp parallel for shared (knn_vec, rnn_vec) private (pointIdxNKNSearch, pointNKNSquaredDistance) num_threads(threads_)
    for (int idx = 0; idx < static_cast<int> (cloud->points.size ()); ++idx)
    {
        if ( kdtree.nearestKSearch (cloud->points[idx], K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
        {
            for(const int nn_idx : pointIdxNKNSearch){
                knn_vec[idx].push_back(nn_idx);
                rnn_vec[nn_idx].push_back(idx);
            }
            knn_dist[idx] = pointNKNSquaredDistance[K-1];
        }else{
            knn_dist[idx] = F_MAX;
        }
    }


    #pragma omp parallel for shared (influential_nn_vec) private (pointIdxNKNSearch, pointNKNSquaredDistance) num_threads(threads_)
    for (int idx = 0; idx < static_cast<int> (cloud->points.size ()); ++idx)
    {
        std::vector<int> knn_idxs = knn_vec[idx];
        std::vector<int> rnn_idxs = rnn_vec[idx];
        
        std::sort(knn_idxs.begin(), knn_idxs.end());
        std::sort(rnn_idxs.begin(), rnn_idxs.end());

        knn_idxs.erase(std::unique(knn_idxs.begin(), knn_idxs.end()), knn_idxs.end());
        rnn_idxs.erase(std::unique(rnn_idxs.begin(), rnn_idxs.end()), rnn_idxs.end());

        std::vector<int> list_inter;
        std::set_intersection(knn_idxs.begin(), knn_idxs.end(), rnn_idxs.begin(), rnn_idxs.end(), std::back_inserter(list_inter));
        influential_nn_vec[idx] = list_inter;
    }

    std::vector<float> inv_k_dist_vec;    
    std::vector<float> den_is_p_vec;
    std::vector<float> inflo_vec;    
    for(int i=0;i<n_points;i++){
        inv_k_dist_vec.push_back(0);
        den_is_p_vec.push_back(0);
        inflo_vec.push_back(0);
    }

    #pragma omp parallel for shared (inv_k_dist_vec) num_threads(threads_)
    for (int idx = 0; idx < static_cast<int> (cloud->points.size ()); ++idx)
    { 
        inv_k_dist_vec[idx] = 1/knn_dist[idx];
    }

    #pragma omp parallel for shared (den_is_p_vec) num_threads(threads_)
    for (int idx = 0; idx < static_cast<int> (cloud->points.size ()); ++idx)
    { 
        influential_nn_vec[idx];
        std::vector<int> influential_nn_ind = influential_nn_vec[idx];
        if(influential_nn_ind.size() > 0){
            float sum_is_density = 0;
            for(const int inn_idx : influential_nn_ind){
                sum_is_density += inv_k_dist_vec[inn_idx];
            }
            den_is_p_vec[idx] = sum_is_density/influential_nn_ind.size();
        }else{
            den_is_p_vec[idx] = F_MAX;
        }
    }

    #pragma omp parallel for shared (inflo_vec) num_threads(threads_)
    for (int idx = 0; idx < static_cast<int> (cloud->points.size ()); ++idx)
    { 
        std::vector<int> neighbor_ind = knn_vec[idx];
        inflo_vec[idx]  = den_is_p_vec[idx]/inv_k_dist_vec[idx];
    }

    inliers->indices.clear();
    outliers->indices.clear();
    for (int idx = 0; idx < static_cast<int> (cloud->points.size ()); ++idx)
    { 
        cout << inflo_vec[idx] << endl;
        if(inflo_vec[idx] <= threshold){
            inliers->indices.push_back(idx);
        }else{
            outliers->indices.push_back(idx);
        }
    }
    end = std::chrono::system_clock::now();  // 計測終了時間
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
    cout << "elapsed:" << elapsed <<  endl;
}