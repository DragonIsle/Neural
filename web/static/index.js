'use strict';

let app = angular.module('home', []);
app.controller('HomeController', function ($scope, $http) {
    const vm = $scope;

    vm.isSend = false;
    vm.isCorrect = true;
    vm.letter = '';

    vm.sendLetter = () => {
        $http.post("/consider", '', {params: {'correctLetter': vm.letter.charAt(0).toUpperCase()}}).then(() => {
            vm.isSend = false;
            vm.isCorrect = true;
        });
    };

    vm.sendPicture = () => {
        let dataURL = document.getElementById('can').toDataURL();
        $.post("/consider/letter",
            {
                imgBase64: dataURL
            },
            function (data) {
                vm.letter = data;
                vm.isSend = true;
                $scope.$apply();
            },
            'text');
    };

    vm.wrongLetter = () => {
        vm.isCorrect = false;
    };

    vm.correctLetter = () => {
        vm.isSend = false
    };
});

