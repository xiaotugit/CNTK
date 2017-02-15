//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <boost/algorithm/string/predicate.hpp>

#include "CNTKLibrary.h"
#include "fileutil.h"
#include "PerformanceProfiler.h"

namespace CNTK
{
    using namespace std;

    const static std::wstring s_trainingMinibatchSource = L"TrainingMinibatchSource";

    inline bool isNumber(const std::wstring& s)
    {
        return !s.empty() &&
            find_if(s.begin(), s.end(), [](wchar_t c) { return !isdigit(c); }) == s.end();
    }

    TrainingSessionPtr CreateBasicTrainingSession(
        const MinibatchSourcePtr& trainingSource,
        const TrainerPtr& trainer,
        const std::unordered_map<Variable, StreamInformation>& modelInputToMinibatchSourceStream,
        const MinibatchSizeSchedule& minibatchSizeSchedule,
        size_t checkpointFrequencyinSamples,
        const std::wstring& checkPointFileName,
        const MinibatchSourcePtr& crossValidationSource,
        const MinibatchSizeSchedule& crossValidationSchedule,
        size_t crossValidationFrequencyInSamples,
        bool restoreFromCheckpointIfExists,
        bool saveAllCheckpoints,
        size_t maxNumberOfSamples,
        size_t progressFrequency)
    {
        fprintf(stderr, "WARNING:CreateBasicTrainingSession is deprecated and will be removed in the next beta (13)."
            "Instructions for updating:"
            "Please switch to CreateTrainingSession function and then call SetCheckpointing/SetCrossValidation/SetPrintingProgress as needed.");

        return MakeSharedObject<TrainingSession>(trainingSource,
            trainer,
            modelInputToMinibatchSourceStream,
            minibatchSizeSchedule,
            checkpointFrequencyinSamples,
            checkPointFileName,
            crossValidationSource,
            crossValidationSchedule,
            crossValidationFrequencyInSamples,
            restoreFromCheckpointIfExists,
            saveAllCheckpoints,
            maxNumberOfSamples,
            progressFrequency);
    }

    TrainingSessionPtr CreateTrainingSession(
        const TrainerPtr& trainer,
        const MinibatchSourcePtr& trainingSource,
        const MinibatchSizeSchedule& minibatchSizeSchedule,
        const std::unordered_map<Variable, StreamInformation>& modelInputToMinibatchSourceStream,
        size_t maxNumberOfSamples)
    {
        return MakeSharedObject<TrainingSession>(
            trainer, 
            trainingSource,
            minibatchSizeSchedule,
            modelInputToMinibatchSourceStream,
            maxNumberOfSamples);
    }

    TrainingSession::TrainingSession(
        const MinibatchSourcePtr& trainingSource,
        const TrainerPtr& trainer,
        const std::unordered_map<Variable, StreamInformation>& modelInputToMinibatchSourceStream,
        const MinibatchSizeSchedule& schedule,
        size_t checkpointFrequencyInSamples,
        const std::wstring& checkPointFileName,
        const MinibatchSourcePtr& crossValidationSource,
        const MinibatchSizeSchedule& crossValidationSchedule,
        size_t crossValidationFrequencyInSamples,
        bool restoreFromCheckpointIfExists,
        bool saveAllCheckpoints,
        size_t maxNumberOfSamples,
        size_t progressFrequencyInSamples) :
        m_trainingSource(trainingSource),
        m_trainer(trainer),
        m_modelInputToMinibatchSourceStream(modelInputToMinibatchSourceStream),
        m_checkPointFileName(checkPointFileName),
        m_parallelAfterSamples(0),
        m_workerRank(0),
        m_numberOfWorkers(1),
        m_minibatchSizeSchedule(schedule),
        m_maxNumberOfSamples(maxNumberOfSamples),
        m_restoreFromCheckpointIfExists(restoreFromCheckpointIfExists),
        m_saveAllCheckpoints(saveAllCheckpoints),
        m_crossValidationSource(crossValidationSource),
        m_crossValidationSchedule(crossValidationSchedule)
    {
        if (!trainingSource)
            InvalidArgument("Training minibatch source is not allowed to be null.");
        if (!trainer)
            InvalidArgument("Trainer is not allowed to be null.");
        if (modelInputToMinibatchSourceStream.empty())
            InvalidArgument("Input mapping is not allowed to be empty.");

        if (m_checkPointFileName.empty())
        {
            if (checkpointFrequencyInSamples != 0 && checkpointFrequencyInSamples != std::numeric_limits<size_t>::max())
                InvalidArgument("Checkpoint file name is not allowed to be empty if checkpoint frequency is non zero.");
            if (saveAllCheckpoints)
                InvalidArgument("Checkpoint file name is not allowed to be empty if 'save all checkpoints' is specified.");
            checkpointFrequencyInSamples = 0;
        }

        // Let's calculate the warm up period the distributed learners may need.
        // We will take the maximum warm up period required.
        auto learners = trainer->ParameterLearners();
        m_parallelAfterSamples = 0;
        for (const auto& l : learners)
        {
            auto distributed = std::dynamic_pointer_cast<DistributedLearner>(l);
            if (distributed)
            {
                m_parallelAfterSamples = std::max(m_parallelAfterSamples, distributed->ParallelizationAfter());
                m_workerRank = distributed->GetCommunicator()->CurrentWorker().m_globalRank;
                m_numberOfWorkers = distributed->GetCommunicator()->Workers().size();
            }
        }

        // Fill-in required actions.
        if (checkpointFrequencyInSamples != 0)
            m_actions.push_back({ checkpointFrequencyInSamples, 0, 0,
                [this](size_t currentIndex, const DeviceDescriptor&)
                {
                    SaveCheckpoint(currentIndex);
                    // enable profiler after the first checkpoint
                    // This has effect only if the profiler is globally enabled by StartProfiler()
                    Microsoft::MSR::CNTK::ProfilerEnable(true);
                    return true;
                } });

        if (crossValidationFrequencyInSamples != 0)
            m_actions.push_back({ crossValidationFrequencyInSamples, 0, 0,
                [this](size_t currentIndex, const DeviceDescriptor& d) { return CrossValidate(currentIndex, d); } });

        if (progressFrequencyInSamples != 0)
            m_actions.push_back({ progressFrequencyInSamples, 0, 0,
                [this](size_t currentIndex, const DeviceDescriptor&) { ReportProgress(currentIndex); return true; } });
    }

    TrainingSession::TrainingSession(
        const TrainerPtr& trainer,
        const MinibatchSourcePtr& trainingSource,
        const MinibatchSizeSchedule& minibatchSizeSchedule,
        const std::unordered_map<Variable, StreamInformation>& modelInputToMinibatchSourceStream,
        size_t maxNumberOfTrainingSamples) :
        m_trainingSource(trainingSource),
        m_trainer(trainer),
        m_modelInputToMinibatchSourceStream(modelInputToMinibatchSourceStream),
        m_parallelAfterSamples(0),
        m_workerRank(0),
        m_numberOfWorkers(1),
        m_minibatchSizeSchedule(minibatchSizeSchedule),
        m_maxNumberOfSamples(maxNumberOfTrainingSamples),
        m_restoreFromCheckpointIfExists(false),
        m_saveAllCheckpoints(false),
        m_crossValidationSchedule(minibatchSizeSchedule)
    {
        if (!trainingSource)
            InvalidArgument("Training minibatch source is not allowed to be null.");
        if (!trainer)
            InvalidArgument("Trainer is not allowed to be null.");
        if (modelInputToMinibatchSourceStream.empty())
            InvalidArgument("Input mapping is not allowed to be empty.");

        // Let's calculate the warm up period the distributed learners may need.
        // We will take the maximum warm up period required.
        auto learners = trainer->ParameterLearners();
        m_parallelAfterSamples = 0;
        for (const auto& l : learners)
        {
            auto distributed = std::dynamic_pointer_cast<DistributedLearner>(l);
            if (distributed)
            {
                m_parallelAfterSamples = std::max(m_parallelAfterSamples, distributed->ParallelizationAfter());
                m_workerRank = distributed->GetCommunicator()->CurrentWorker().m_globalRank;
                m_numberOfWorkers = distributed->GetCommunicator()->Workers().size();
            }
        }
    }

    TrainingSession& TrainingSession::WithCheckpointing(
        const std::wstring& checkPointFileName,
        size_t checkpointFrequencyInSamples,
        bool restoreFromCheckpointIfExists,
        bool saveAllCheckpoints)
    {
        m_checkPointFileName = checkPointFileName;
        if (m_checkPointFileName.empty())
        {
            if (checkpointFrequencyInSamples != 0 && checkpointFrequencyInSamples != std::numeric_limits<size_t>::max())
                InvalidArgument("Checkpoint file name is not allowed to be empty if checkpoint frequency is non zero.");
            if (saveAllCheckpoints)
                InvalidArgument("Checkpoint file name is not allowed to be empty if 'save all checkpoints' is specified.");
            checkpointFrequencyInSamples = 0;
        }

        m_saveAllCheckpoints = saveAllCheckpoints;
        m_restoreFromCheckpointIfExists = restoreFromCheckpointIfExists;

        // Fill-in required actions.
        if (checkpointFrequencyInSamples != 0)
            m_actions.push_back({ checkpointFrequencyInSamples, 0, 0,
                [this](size_t currentIndex, const DeviceDescriptor&)
        {
            SaveCheckpoint(currentIndex);
            // enable profiler after the first checkpoint
            // This has effect only if the profiler is globally enabled by StartProfiler()
            Microsoft::MSR::CNTK::ProfilerEnable(true);
            return true;
        } });

        return *this;
    }

    ///
    /// Sets cross validation configuration.
    /// crossValidationSource: a minibatch source that will be used for cross validation.
    /// crossValidationSchedule : a minibatch size schedule for cross validation.
    ///
    TrainingSession& TrainingSession::WithCrossValidation(
        const MinibatchSourcePtr& crossValidationSource,
        const MinibatchSizeSchedule& crossValidationSchedule,
        size_t crossValidationFrequencyInSamples)
    {
        m_crossValidationSource = crossValidationSource;
        m_crossValidationSchedule = crossValidationSchedule;

        if (crossValidationFrequencyInSamples != 0)
            m_actions.push_back({ crossValidationFrequencyInSamples, 0, 0,
                [this](size_t currentIndex, const DeviceDescriptor& d) { return CrossValidate(currentIndex, d); } });

        return *this;
    }

    TrainingSession& TrainingSession::WithProgressPrinting(size_t progressFrequency)
    {
        if (progressFrequency != 0)
            m_actions.push_back({ progressFrequency, 0, 0,
                [this](size_t currentIndex, const DeviceDescriptor&) { ReportProgress(currentIndex); return true; } });

        return *this;
    }

    void TrainingSession::Train(const DeviceDescriptor& computeDevice)
    {
        std::unordered_map<Variable, ValuePtr> minibatch;
        bool shouldTrain = m_maxNumberOfSamples > 0;

        // Let's try to restore if required.
        size_t restoredNumberOfSamples = 0;
        if (m_restoreFromCheckpointIfExists && !m_checkPointFileName.empty())
        {
            RestoreFromCheckpoint();
            restoredNumberOfSamples = m_trainer->TotalNumberOfSamplesSeen();
        }

        // Main train loop.
        bool earlyExit = false;
        while (shouldTrain)
        {
            // Get next minibatch.
            size_t samplesLeft = earlyExit || m_maxNumberOfSamples <= m_trainer->TotalNumberOfSamplesSeen() 
                ? 0
                : m_maxNumberOfSamples - m_trainer->TotalNumberOfSamplesSeen();

            GetTrainingMinibatch(minibatch, samplesLeft, computeDevice);

            // Train on the minibatch
            OnMinibatchStart();
            shouldTrain = m_trainer->TrainMinibatch(minibatch, computeDevice);
            OnMinibatchEnd();

            auto profMisc = Microsoft::MSR::CNTK::ScopeProfile(Microsoft::MSR::CNTK::profilerEvtMainPost);

            // Peform actions if required.
            size_t totalNumberOfSamples = m_trainer->TotalNumberOfSamplesSeen();
            for (auto& action : m_actions)
            {
                size_t index = totalNumberOfSamples / action.frequency;
                if (index != action.currentIndex)
                {
                    bool shouldContinue = action.action(action.currentIndex, computeDevice);
                    if (!shouldContinue) // If any action wants to have early exit - we stop training.
                        earlyExit = true;

                    action.currentIndex = index;
                    action.sampleCountWhenLastCalled = totalNumberOfSamples;
                }
            }
        }

        if (restoredNumberOfSamples != m_trainer->TotalNumberOfSamplesSeen())
        {
            // Let's do all actions on the last probably a partial data at the end.
            for (auto& action: m_actions)
            {
                if (m_trainer->TotalNumberOfSamplesSeen() % action.frequency != 0 &&
                    m_trainer->TotalNumberOfSamplesSeen() != action.sampleCountWhenLastCalled)
                    action.action(action.currentIndex, computeDevice);
            }
        }

        // In case of incremental - save final checkpoint.
        // This is required only when we keep all existing checkpoints, otherwise 
        // The checkpoint was already saved with the proper name.
        if (m_saveAllCheckpoints && !fexists(m_checkPointFileName))
            SaveFinalCheckpoint();
    }

    // TODO: Possibly expose a limiting counter on the number of samples for validation.
    bool TrainingSession::CrossValidate(size_t currentIndex, const DeviceDescriptor& computeDevice)
    {
        if (m_crossValidationSource) // Running cross validation
        {
            std::unordered_map<Variable, ValuePtr> minibatch;
            double accumulatedError = 0;
            double error;
            size_t totalNumberOfSamples = 0;
            size_t numberOfMinibatches = 0;

            auto checkpoint = m_crossValidationSource->GetCheckpointState();
            size_t sampleCount = 0;
            while (GetCrossValidationMinibatch(minibatch, m_crossValidationSchedule[sampleCount], computeDevice), !minibatch.empty())
            {
                error = m_trainer->TestMinibatch(minibatch, computeDevice, sampleCount);
                accumulatedError += error;
                totalNumberOfSamples += sampleCount;
                numberOfMinibatches++;
            }

            m_crossValidationSource->RestoreFromCheckpoint(checkpoint);
            return OnCrossValidationEnd(currentIndex, accumulatedError / totalNumberOfSamples, totalNumberOfSamples, numberOfMinibatches);
        }
        else // Only invoking the callback.
        {
            return OnCrossValidationEnd(currentIndex, 0, 0, 0);
        }
    }

    inline void TrainingSession::ReportProgress(size_t currentIndex)
    {
        this->OnProgress(currentIndex);
    }

    void TrainingSession::GetTrainingMinibatch(std::unordered_map<Variable, ValuePtr>& minibatch, size_t maxMbSize, const DeviceDescriptor& computeDevice)
    {
        size_t workerRank = m_workerRank, numberOfWorkers = m_numberOfWorkers;

        // Check if we are operating in distributed mode.
        if (m_parallelAfterSamples > m_trainer->TotalNumberOfSamplesSeen())
        {
            numberOfWorkers = 1;
            workerRank = 0;
        }

        size_t mbSize = GetMinibatchSize();
        mbSize = std::min(mbSize, maxMbSize);
        GetNextMinibatch(m_trainingSource, minibatch, mbSize, workerRank, numberOfWorkers, computeDevice);
    }

    void TrainingSession::GetCrossValidationMinibatch(std::unordered_map<Variable, ValuePtr>& minibatch, size_t maxMbSize, const DeviceDescriptor& computeDevice)
    {
        // TODO: Support distributed cross-validation, when TestMinibatch supports it.
        GetNextMinibatch(m_crossValidationSource, minibatch, maxMbSize, 0, 1, computeDevice);
    }

    void TrainingSession::GetNextMinibatch(const MinibatchSourcePtr& source, std::unordered_map<Variable, ValuePtr>& minibatch, size_t mbSize, size_t workerRank, size_t numberOfWorkers, const DeviceDescriptor& computeDevice)
    {
        minibatch.clear();

        if (mbSize == 0)
            return;

        auto minibatchData = source->GetNextMinibatch(0 /*numberOfSequences*/, mbSize, numberOfWorkers, workerRank, computeDevice);
        if (minibatchData.empty())
            return;

        for (auto v : m_modelInputToMinibatchSourceStream)
            minibatch.insert({ v.first, minibatchData[v.second].data });
    }

    void TrainingSession::RestoreFromCheckpoint(const std::wstring& checkpointFileName)
    {
        Dictionary externalState = m_trainer->RestoreFromCheckpoint(checkpointFileName);
        m_trainingSource->RestoreFromCheckpoint(externalState[s_trainingMinibatchSource].Value<Dictionary>());
    }

    void TrainingSession::SaveCheckpoint(size_t currentIndex)
    {
        OnCheckpointStart(currentIndex);
        Dictionary externalState;
        externalState[s_trainingMinibatchSource] = m_trainingSource->GetCheckpointState();

        wstring checkpointFile = m_checkPointFileName;
        if (m_saveAllCheckpoints)
            checkpointFile += std::to_wstring(currentIndex);
        m_trainer->SaveCheckpoint(checkpointFile, externalState);
        OnCheckpointEnd(currentIndex);
    }

    void TrainingSession::SaveFinalCheckpoint()
    {
        Dictionary externalState;
        externalState[s_trainingMinibatchSource] = m_trainingSource->GetCheckpointState();
        m_trainer->SaveCheckpoint(m_checkPointFileName, externalState);
    }

    // Restores from a m_checkPointFileName file.
    // If the file path exists - simply restores from the corresponding file.
    // If the file path does not exist - looks into directory where the file is
    // located and picks up the file with the largest N among <m_checkPointFileName>N files,
    // Where N is some positive integer.
    void TrainingSession::RestoreFromCheckpoint()
    {
        assert(!m_checkPointFileName.empty());

        // Make sure the intermediate directories exist, so no need for further checks.
        msra::files::make_intermediate_dirs(m_checkPointFileName);

        size_t pos = m_checkPointFileName.find_last_of(L"\\/");
        wstring parent;
        wstring fileName;
        if (pos == wstring::npos)
        {
            parent = L"..";
            fileName = m_checkPointFileName;
        }
        else
        {
            parent = m_checkPointFileName.substr(0, pos);
            fileName = m_checkPointFileName.substr(pos);
        }

        std::wstring restoreFile;
        if (fexists(m_checkPointFileName))
        {
            restoreFile = m_checkPointFileName;
        }
        else
        {
            // let's check whether there are other possible candidates to restore from.
            int maxValue = -1;
            std::vector<std::wstring> files = msra::files::get_all_files_from_directory(parent);

            for (auto f : files)
            {
                if (!boost::starts_with(f, fileName))
                {
                    continue;
                }

                auto suffix = f.substr(fileName.size());
                if (!isNumber(suffix) || !fexists(parent + L"/" + f + L".ckp"))
                {
                    continue;
                }

                auto expectedNumber = msra::strfun::utf8(suffix);
                char* tmp = nullptr;
                int value = strtol(expectedNumber.c_str(), &tmp, 10);
                if (tmp != expectedNumber.c_str() + expectedNumber.size())
                    continue;

                if (value > maxValue)
                {
                    // Found a better candidate.
                    maxValue = value;
                    restoreFile = parent + L"/" + f;
                }
            }
        }

        if (restoreFile.empty()) // Nothing to restore.
            return;

        // TODO: Should have proper loggin instead.
        fprintf(stderr, "Restoring training session from the checkpoint '%ls'\n", restoreFile.c_str());

        this->RestoreFromCheckpoint(restoreFile);

        // Recalculate actions indicies.
        size_t totalNumberOfSamples = m_trainer->TotalNumberOfSamplesSeen();
        for (auto& action : m_actions)
        {
            action.currentIndex = totalNumberOfSamples / action.frequency;
            action.sampleCountWhenLastCalled = totalNumberOfSamples - totalNumberOfSamples % action.frequency;
        }
    }
}
