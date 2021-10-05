#include <string>
#include <iostream>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <pthread.h>
#include <ros/ros.h>
#include <std_msgs/String.h>

using boost::asio::ip::udp;

namespace {

class UDPServer {
public:
    ros::NodeHandle nh;
    ros::Publisher pub = nh.advertise<std_msgs::String>("/robot_news_radio",10);

    UDPServer(boost::asio::io_service& io_service)
        : _socket(io_service, udp::endpoint(udp::v4(), 8182))
    {
        startReceive();
    }
private:
    void startReceive() {
        _socket.async_receive_from(
            boost::asio::buffer(_recvBuffer), _remoteEndpoint,
            boost::bind(&UDPServer::handleReceive, this,
                boost::asio::placeholders::error,
                boost::asio::placeholders::bytes_transferred));
    }

    void handleReceive(const boost::system::error_code& error,
                       std::size_t bytes_transferred) {
        if (!error || error == boost::asio::error::message_size) {
            // std::cout << "Received message" << std::string( std::begin(_recvBuffer), bytes_transferred ) << std::endl;
            // std::cout << std::string( std::begin(_recvBuffer), bytes_transferred)  << std::endl;
            std_msgs::String msg;
            msg.data = std::string( std::begin(_recvBuffer), bytes_transferred);
            pub.publish(msg);

            // auto message = std::make_shared<std::string>("Hello, World\n");

            // _socket.async_send_to(boost::asio::buffer(*message), _remoteEndpoint,
            //     boost::bind(&HelloWorldServer::handleSend, this, message,
            //         boost::asio::placeholders::error,
            //         boost::asio::placeholders::bytes_transferred));
            startReceive();
        }
    }

    void handleSend(std::shared_ptr<std::string> message,
                    const boost::system::error_code& ec,
                    std::size_t bytes_transferred) {
        startReceive();
    }

    udp::socket _socket;
    udp::endpoint _remoteEndpoint;
    std::array<char, 1024> _recvBuffer;
};

}  // namespace

int main(int argc,char **argv) {
    ros::init(argc,argv,"robot_news_radio_cpp");

    while (ros::ok()){
        try {
            boost::asio::io_service io_service;
            UDPServer server{io_service};
            
            io_service.run();
            
        } catch (const std::exception& ex) {
            std::cerr << ex.what() << std::endl;
        }
    }
    return 0;
}


/*
int main (int argc,char **argv)
{
    ros::init(argc,argv,"robot_news_radio_cpp");
    ros::NodeHandle nh;

    ros::Publisher pub = nh.advertise<std_msgs::String>("/robot_news_radio",10);

    //ROS_INFO("Node has been started");

    //ros::Duration(1.0).sleep();
    ros::Rate rate(3);

    while (ros::ok()){
        std_msgs::String msg;
        msg.data = "Hi,it is good to see you!";
        pub.publish(msg);
        rate.sleep();
    }

}
*/