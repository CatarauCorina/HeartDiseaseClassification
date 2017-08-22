'use strict';   // See note about 'use strict'; below

var myApp = angular.module('myApp', [
 'ngRoute',
]);

myApp.config(['$routeProvider',
     function($routeProvider) {
         $routeProvider.
             when('/', {
                 templateUrl: '/static/partials/index.html',
             }).
             when('/about', {
                 templateUrl: '../static/partials/about.html',
             }).
             otherwise({
                 redirectTo: '/'
             });
    }]);

 
myApp.controller('HeartController', function($scope, $http) {
      $scope.master = {};
      
      
   $scope.formData = {};

 $scope.sendData = function(user){
    $http({
        method: "POST",
        url: "/sendUser",
        headers: {
            'Access-Control-Allow-Origin': '*',
            'Content-Type': 'application/json',
        },
        data: {
            'user':{user},
        }
    }).then(function successCallback(response){
        console.log(response.data)
        $scope.message = "";
    });
}

      $scope.update = function(user) {
        
        $scope.master = angular.copy(user);
      };

      $scope.reset = function() {
        $scope.user = angular.copy($scope.master);
      };

      $scope.reset();
    });