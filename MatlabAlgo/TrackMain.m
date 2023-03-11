function TrackMain

% You need to have the videos on which you work in the same file as this algorithm.
%
% STEPS TO USING ADJUSTING THE ALGORITHM TO NEW VIDEOS:
%
% 1. COPING THE ENTIRE "Parameters" PARAGRAPH INTO THE COMMAND WINDOW' AND
% PRESS ENTER (DONT FORGET TO CHANGE THE FILE NAME TO THE VIDEO NAME)
%
% 2. COPY THE FOLLOWING PARAGRAPH INTO THE COMMAND WINDOW AND PRESS ENTER
%
%           VideoFile = VideoReader(MovieFileName);
%           Frames = read(VideoFile, [1 10]);
%           imshow (Frames (:,:,:,1))
%
%    AND CHOOSE THE CORRECT PARAMETERS FOR THE TOP, BOTTOM, LEFT AND RIGHT PARAMETERS.
%    MAKE SURE THAT THE LEFT PARAMETER IS AFTER THIS SHINY ROD THINGY IN
%    THE RIGHT SIDE, BECAUSE THE ALOGORITHM DOESNT KNOW HOW TO HANDLE IT.
%
% 3. NOW, YOU NEED TO FIND 10 COSECUTIVE FRAMES IN WHICH THERE IS NO DROP
% IN THE FRAME. IT WILL BE MOST LIKELY IN THE FIRST 10 FRAMES, BUT IF NOT
% JUST GO IN STEPS OF 50 FRAMES AND FIND 10 FRAMES WHICH DONT HAVE ANY
% DROPS IN THEM.
%
% 4. PUT THE NUMBER OF THE 10TH FRAME IN THE "BackgroundNumOfFrames"
% PARAMETER.
%

%% Parameters
MovieFileName = "250.avi";
BackgroundNumOfFrames = 450;
Top = 50;    % In these lines insert the correct
Bottom = 263; % indexes. it should be the same if its
Left = 625;   % two videos from the same lab, but if
Right = 1780; % its not, it needs to be recalibrated.
DiffTH = 40;
Filter = ones(25,25);
ClustTH = 225;
ActivationFlag1 = 0;
ActivationFlag2 = false;
DropletNum = 0;
LabNum = "lab6";
PixelToMMConvertion = 34/188; % Might change for different videos, depends if the camera has been moved.

%% Background setting, Frame rate setting and Total number of Frames Setting
VideoObj = VideoWriter('310-9-lab2-Filtered.avi');
open(VideoObj);
VideoFile = VideoReader(MovieFileName);
Frames = read(VideoFile,[1 BackgroundNumOfFrames]);
Background = uint8(mean(Frames(Top:Bottom,Left:Right,1,BackgroundNumOfFrames-9:BackgroundNumOfFrames),4));
FrameRate = VideoFile.FrameRate;
NumOfFrames = floor(FrameRate*VideoFile.Duration);
%%%DropSize = zeros(size(NumOfFrames));

%% Setting time axis
for i=1:NumOfFrames
    tAxis(i) = i/FrameRate;
    CenterR(i) = 0;
    %%%DropRad(i) = 0;
    %%%Rad(i) = 0;
end

%% AutoTracker Algorithm
for FrameNumber=6:NumOfFrames
    Frame = read(VideoFile,FrameNumber);
    CurrentImage = Frame(Top:Bottom,Left:Right,1);
    DiffImage = abs(CurrentImage-Background);
    BinImage = zeros(size(DiffImage));
    BinImage(find(DiffImage>DiffTH))=1;
    ConvImage = conv2(BinImage,Filter);
    MaxDrop = max(max(ConvImage));
    FrameNumber
    if MaxDrop > 0
        ClustImage = zeros(size(ConvImage));
        Ind = find(ConvImage == MaxDrop);
        ClustImage(Ind)=255;
        writeVideo(VideoObj,uint8(ClustImage));
        SumHor = sum(ClustImage, 2);
        SumVert = sum(ClustImage,1);
        MaxQuadHorEdge = max(SumHor);
        MaxQuadVertEdge = max(SumVert);
        QuadIndR(FrameNumber) = mean(find(SumHor == MaxQuadHorEdge));
        QuadIndC(FrameNumber) = mean(find(SumVert == MaxQuadVertEdge));
        %%%UpStep = QuadIndR(FrameNumber);
        %%%DownStep = QuadIndR(FrameNumber);
        %%%LeftStep = QuadIndC(FrameNumber);
        %%%RightStep = QuadIndC(FrameNumber);
        %%%UpRad = 0;
        %%%DownRad = 0;
        %%%LeftRad = 0;
        %%%RightRad = 0;
        %%%[NumOfRows, NumOfCols] = size(ClustImage);
        %%%if QuadIndR(FrameNumber)+UpStep >=1 && QuadIndR(FrameNumber)+DownStep <= NumOfRows && QuadIndC(FrameNumber)+LeftStep >= 1 && QuadIndC(FrameNumber)+RightStep <= NumOfCols
            %%%while ClustImage(floor(QuadIndR(FrameNumber))+floor(UpStep),floor(QuadIndC(FrameNumber))) == 255 || ClustImage(floor(QuadIndR(FrameNumber))+floor(DownStep),floor(QuadIndC(FrameNumber))) == 255 || ClustImage(floor(QuadIndR(FrameNumber)),floor(QuadIndC(FrameNumber))+floor(LeftStep)) == 255 || ClustImage(floor(QuadIndR(FrameNumber)),floor(QuadIndC(FrameNumber))+floor(RightStep)) == 255
                %%%if UpStep>=2 && ClustImage(UpStep,QuadIndC) == 255
                    %%%UpStep = UpStep - 1;
                    %%%UpRad =+ 1;
                %%%end
                %%%if DownStep<NumOfRows && ClustImage(DownStep,QuadIndC) == 255
                    %%%DownStep = DownStep+1;
                    %%%DownRad =+ 1;
                %%%end
                %%%if LeftStep>=2 && ClustImage(QuadIndR,LeftStep) == 255
                    %%%LeftStep =- 1;
                    %%%LeftRad =+ 1;
                %%%end
                %%%if RightStep<=NumOfCols && ClustImage(QuadIndR,RightStep) == 255
                    %%%RightStep =+ 1;
                    %%%RightRad =+ 1;
                %%%end
                %%%Rad(FrameNumber) = mean([UpRad, RightRad, LeftRad, DownRad]);
            %%%end
        %%%else
            %%%Rad(FrameNumber) = 0;
        %%%end

        if ActivationFlag2 == false
            for i=0:2
                if QuadIndR(FrameNumber-i)-QuadIndR(FrameNumber-i-1) < 0
                    ActivationFlag1 = ActivationFlag1 + 1;
                end
            end
        end
        if ActivationFlag1 == 3
            if ActivationFlag2 == true
                CenterR(FrameNumber) = QuadIndR(FrameNumber);
                %%%DropRad(FrameNumber) = Rad(FrameNumber);
            else
                for i=0:2
                    CenterR(FrameNumber-i) = QuadIndR(FrameNumber-i);
                    %%%DropRad(FrameNumber-i) = Rad(FrameNumber-i);
                end
                ActivationFlag2 = true;
                DropletNum = DropletNum + 1;
                DropletStart(DropletNum) = FrameNumber;
            end
        else
            CenterR(FrameNumber) = 0;
            %%%DropRad(FrameNumber) = 0;
        end
     else
        if ActivationFlag2 == true
            DropletEnd(DropletNum) = FrameNumber-5;
        end
        ActivationFlag1 = 0;
        ActivationFlag2 = false;
    end
end
if MaxDrop ~= 0
   DropletEnd(DropletNum) = FrameNumber;
end

%% Optimising the Y Axes
YOrigin = max(CenterR)+20;
for i = 1:NumOfFrames
    CenterR(i) = (-1)*(CenterR(i) - YOrigin)*PixelToMMConvertion;
    if CenterR(i) == YOrigin*PixelToMMConvertion
        CenterR(i) = 0;
    end
end

%% Optimising the R Axes
%%%for i = 1:NumOfFrames
    %%%DropRad(i) = DropRad(i)*PixelToMMConvertion;
%%%end

%% Exporting Data to a CSV file to the same file as the Algorithm
for i=1:DropletNum
    writematrix(["t" "y";transpose([tAxis(DropletStart(i):DropletEnd(i)); CenterR(DropletStart(i):DropletEnd(i))])],LabNum+"-"+extractBefore(MovieFileName,6)+"-"+char(i+64)+".csv");
    %%%writematrix(["t" "r";transpose([tAxis(DropletStart(i):DropletEnd(i)); DropRad(DropletStart(i):DropletEnd(i))])],LabNum+"-"+extractBefore(MovieFileName,6)+"-"+char(i+64)+"-r.csv");
end

for i=1:DropletNum
    NewPlot(i) = nexttile;
    plot(NewPlot(i),tAxis(DropletStart(i):DropletEnd(i)),CenterR(DropletStart(i):DropletEnd(i)))
end

close(VideoObj);

%%%for i=1:DropletNum
    %%%NewPlotR(i) = nexttile;
    %%%plot(NewPlotR(i),tAxis(DropletStart(i):DropletEnd(i)),DropRad(DropletStart(i):DropletEnd(i)))

end
%hold off
%plot(tAxis,CenterR)
